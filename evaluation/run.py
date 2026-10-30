#!/usr/bin/env python3
"""Reproducible, provider-free evaluation against the configured historical IPC PDF."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
# This pilot is explicitly offline: never call a configured generative provider.
os.environ['OPENAI_API_KEY'] = ''
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from app.config import get_settings
from app.services import chatbot_service, retrieval_service, source_service

METHODS = ('bm25', 'semantic', 'hybrid')
K = 5


def contains_label(text, section):
    return bool(re.search(r'\[s\s+' + str(section) + r'\]', text, flags=re.I))


def label_validation(corpus, dataset):
    if corpus.pdf_hash != dataset['corpus_sha256']:
        raise ValueError('PDF hash does not match the labelled corpus; do not score against a different version.')
    for case in dataset['cases']:
        if 'section' not in case:
            continue
        for page in case['pages']:
            if not any(p == page and contains_label(t, case['section']) for p, t in zip(corpus.pages, corpus.texts)):
                raise ValueError(f"{case['id']}: section marker missing from labelled page {page}")


def semantic_baseline(query, corpus):
    """Standalone dense-only baseline; this is not a production API mode."""
    start = perf_counter()
    vectors = retrieval_service._vectors(corpus)
    vector = np.asarray(retrieval_service._model().encode([query], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)[0]
    scores = vectors @ vector
    ranked = [int(i) for i in np.argsort(-scores, kind='stable')[:60] if scores[i] >= retrieval_service.MIN_DENSE_SIMILARITY]
    # Match the production hybrid's explicit provision-number constraint.
    match = re.search(r'\b(?:section|sec|s\.)\s*(\d+[a-z]*)\b', query, flags=re.I)
    if match:
        ranked = [i for i in ranked if match.group(1).casefold() in corpus.tokens[i]]
    return {'results': [{'page': corpus.pages[i], 'excerpt': corpus.texts[i], 'id': corpus.ids[i]} for i in ranked[:K]],
            'retrieval': {'method': 'semantic_baseline', 'warning': None}, 'elapsed_ms': (perf_counter() - start) * 1000}


def run_method(method, query, corpus):
    if method == 'semantic':
        return semantic_baseline(query, corpus)
    started = perf_counter()
    result = retrieval_service.retrieve(query, domain='ipc', top_k=K, method=method)
    result['elapsed_ms'] = (perf_counter() - started) * 1000
    return result


def score(case, response):
    found = None
    for rank, item in enumerate(response['results'], 1):
        if item['page'] in case['pages'] and contains_label(item['excerpt'], case['section']):
            found = rank
            break
    return {'hit': found is not None, 'rank': found, 'reciprocal_rank': 1 / found if found else 0.0,
            'top_pages': [item['page'] for item in response['results']],
            'elapsed_ms': round(response['elapsed_ms'], 2), 'actual_method': response['retrieval']['method'],
            'warning': response['retrieval'].get('warning')}


def percentile(values, percentile_value):
    return round(float(np.percentile(values, percentile_value)), 2) if values else None


def summarize_method(results, method):
    expected={'bm25':'bm25','semantic':'semantic_baseline','hybrid':'hybrid'}[method]
    returned=[r for r in results if not r.get('unavailable')]
    actual=[r for r in returned if r['actual_method']==expected]
    times=[r['elapsed_ms'] for r in actual]
    return {'hits':sum(r['hit'] for r in actual),'total':len(actual),
            'hit_rate':sum(r['hit'] for r in actual)/len(actual) if actual else 0,
            'mrr':statistics.mean(r['reciprocal_rank'] for r in actual) if actual else 0,
            'p50_ms':percentile(times,50),'p95_ms':percentile(times,95),
            'fallbacks':len(returned)-len(actual),'unavailable':len(results)-len(returned)}


def render_markdown(result):
    out = ['# Historical IPC retrieval and scope evaluation', '',
           '> Historical research material only. This is not current-law advice, legal applicability, or a legal correctness benchmark.', '',
           f"- Corpus SHA-256: `{result['corpus_sha256']}`", f"- Executed (UTC): {result['executed_at']}",
           f"- Source: {result['source_title']} ({result['corpus_pages']} physical PDF pages)",
           f"- Model: `{result['model']}`; chunking: {result['chunking']}", '',
           '**Label quality:** Author-created, source-checked pilot questions. They are not sampled from real users, independently adjudicated, or validated by legal experts. Labels contain selected physical pages and exact section markers; other relevant pages may be unlabelled.', '',
           '**What is measured:** Hit@5 and MRR@5 count retrieval of a labelled page *and* matching section marker. This is an evidence-location proxy, not proof that an answer is correct. Latency is warm in-process wall-clock time on this machine; it excludes one-time indexing/model loading and is not a deployment SLA.', '',
           'The semantic-only method is an evaluation baseline using the same local MiniLM encoder and threshold, not an existing standalone product feature. BM25 and hybrid call the actual application service. Hybrid may use a BM25 fallback; fallbacks are shown below but excluded from hybrid metrics and their denominator.', '',
           '| Method | Label hit@5 | MRR@5 | p50 ms | p95 ms | Fallbacks / unavailable |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in METHODS:
        s = result['summary'][name]
        out.append(f"| {name} | {s['hits']}/{s['total']} ({s['hit_rate']:.1%}) | {s['mrr']:.3f} | {s['p50_ms']} | {s['p95_ms']} | {s['fallbacks']} / {s['unavailable']} |")
    out += ['', '### Pilot breakdown', '', '| Question type | BM25 hits | Semantic hits | Hybrid hits |', '| --- | ---: | ---: | ---: |']
    for group, label in [('section_lookup', 'Explicit section lookup'), ('paraphrase', 'Paraphrase')]:
        cases=[c for c in result['supported'] if c['group']==group]
        cells=[]
        for method in METHODS:
            group_summary=summarize_method([c['methods'][method] for c in cases],method)
            cells.append(f"{group_summary['hits']}/{group_summary['total']}" if group_summary['total'] else 'unavailable')
        out.append(f"| {label} | {' | '.join(cells)} |")
    out += ['', '## Supported historical questions', '', 'A hit means the labelled marker is found in the top five retrieved passages. A miss is not automatically a wrong legal answer.', '',
            '| ID | Question | Gold section / pages | BM25 | Semantic baseline | Hybrid |', '| --- | --- | --- | --- | --- | --- |']
    for case in result['supported']:
        def cell(name):
            x=case['methods'][name]
            if x.get('unavailable'):return 'Unavailable'
            return ('hit rank '+str(x['rank']) if x['hit'] else 'miss') + f"; {x['elapsed_ms']} ms; pages {', '.join(map(str,x['top_pages'])) or 'none'}" + ('; fallback' if x['actual_method'] != {'bm25':'bm25','semantic':'semantic_baseline','hybrid':'hybrid'}[name] else '')
        out.append(f"| {case['id']} | {case['query']} | {case['section']} / {', '.join(map(str,case['pages']))} | {cell('bm25')} | {cell('semantic')} | {cell('hybrid')} |")
    out += ['', '## Out-of-scope questions', '', 'These test the actual chat service in provider-free excerpt mode. A pass means the service returned `no_evidence`; it does not establish legal correctness.', '',
            '| ID | Question | Expected | Actual | Outcome | Reason | ms |', '| --- | --- | --- | --- | --- | --- | ---: |']
    for case in result['unsupported']:
        reason=(case.get('reason') or 'None').replace('|','/').replace('\n',' ')
        out.append(f"| {case['id']} | {case['query']} | no_evidence | {case['actual_mode']} | {'pass' if case['passed'] else 'FAIL'} | {reason} | {case['elapsed_ms']} |")
    out += ['', '## Actual excerpt responses', '',
            'The actual chat service was run in provider-free excerpt mode with BM25. Unlike the retrieval table above, the chat service retrieves up to **six** passages, so its hits are not directly comparable to Hit@5. A marker hit means the response included the labelled section on a labelled page. It does not determine whether the passage fully answers the question. Misses and `no_evidence` outputs remain visible.', '',
            '| ID | Actual mode | Labelled evidence in response | Source pages | Chat ms |', '| --- | --- | --- | --- | ---: |']
    for case in result['supported']:
        a=case['chat']
        out.append(f"| {case['id']} | {a['mode']} | {'hit' if a['labelled_evidence'] else 'MISS'} | {', '.join(map(str,a['pages'])) or 'none'} | {a['elapsed_ms']} |")
    out += ['', '## Answer-quality limits', '',
            'Generated answers were **not evaluated**: no generative provider was called. Page-and-section hits and citation membership do not establish that a generated claim is supported or legally correct. The cases are a small development pilot, not a representative or held-out legal benchmark. Ask an independent legal reviewer to annotate answer correctness, evidence support, and appropriate abstention before publishing answer-quality percentages.', '',
            '## Reproduce', '', 'From the repository root, install the locked backend requirements, ensure the configured MiniLM model is cached locally, then run:', '', '```bash', 'python evaluation/run.py', '```', '',
            'The runner validates the source hash and every labelled section marker. It records requested and actual retrieval methods, returns unsupported-question outcomes, and writes `results.json` and this report. Model or corpus failures are never counted as successful semantic results.', '']
    return '\n'.join(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'evaluation'/'results')
    args=parser.parse_args()
    dataset=json.loads((ROOT/'evaluation'/'questions.json').read_text(encoding='utf-8'))
    corpus=retrieval_service._get_corpus('ipc')
    label_validation(corpus,dataset)
    # Warm the actual application caches; setup costs are explicitly excluded.
    retrieval_service.retrieve('historical IPC section 302 murder', 'ipc', top_k=K, method='bm25')
    model_error=None
    try:
        retrieval_service._vectors(corpus)
        retrieval_service._model().encode(['warmup'], normalize_embeddings=True, show_progress_bar=False)
    except Exception as e:
        model_error=f'{type(e).__name__}: local model or embeddings unavailable'
    supported=[]
    for case in dataset['cases']:
        if 'section' not in case:continue
        entry={k:case[k] for k in ('id','group','query','section','pages')}
        entry['methods']={}
        for method in METHODS:
            if method=='semantic' and model_error:
                entry['methods'][method]={'unavailable':model_error}
                continue
            try:
                entry['methods'][method]=score(case,run_method(method,case['query'],corpus))
            except Exception as e:
                entry['methods'][method]={'unavailable':type(e).__name__}
        chat_start=perf_counter()
        try:
            answer=chatbot_service.chat('ipc',case['query'],method='bm25',answer_mode='excerpts')
            answer_score=score(case,{'results':answer['sources'],'retrieval':answer['retrieval'],'elapsed_ms':0})
            marker_in_response=contains_label(answer['response'],case['section'])
            entry['chat']={'mode':answer['mode'],'labelled_evidence':answer_score['hit'] and marker_in_response,
                           'pages':[source['page'] for source in answer['sources']],
                           'reason':answer.get('reason'),'elapsed_ms':round((perf_counter()-chat_start)*1000,2)}
        except Exception as e:
            entry['chat']={'mode':'error','labelled_evidence':False,'pages':[],
                           'reason':type(e).__name__,'elapsed_ms':round((perf_counter()-chat_start)*1000,2)}
        supported.append(entry)
        print('Evaluated',case['id'],flush=True)
    unsupported=[]
    for case in dataset['cases']:
        if 'expected' not in case:continue
        start=perf_counter()
        try:
            response=chatbot_service.chat('ipc',case['query'],method='bm25',answer_mode='excerpts')
            mode=response['mode']
            reason=response.get('reason')
        except Exception as e:
            mode='error'
            reason=type(e).__name__
        unsupported.append({'id':case['id'],'group':case['group'],'query':case['query'], 'actual_mode':mode,
                            'reason':reason,'passed':mode==case['expected'],'elapsed_ms':round((perf_counter()-start)*1000,2)})
    summary={}
    for method in METHODS:
        summary[method]=summarize_method([c['methods'][method] for c in supported],method)
    result={'dataset':dataset['name'],'label_status':dataset['label_status'],'executed_at':datetime.now(timezone.utc).isoformat(),
            'corpus_sha256':corpus.pdf_hash,'corpus_pages':source_service.metadata(corpus.document)[0],
            'source_title':corpus.document.title,'corpus_status':corpus.document.corpus_status,
            'model':get_settings().EMBEDDING_MODEL,'chunking':retrieval_service.CACHE_VERSION,
            'model_error':model_error,'summary':summary,'supported':supported,'unsupported':unsupported,
            'generated_answer_quality':'not evaluated; no provider called','method_order':list(METHODS)}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (args.out_dir/'REPORT.md').write_text(render_markdown(result),encoding='utf-8')
    print(json.dumps({'summary':summary,'unsupported_passed':sum(x['passed'] for x in unsupported),'unsupported_total':len(unsupported),'output':str(args.out_dir)},indent=2))


if __name__=='__main__':main()
