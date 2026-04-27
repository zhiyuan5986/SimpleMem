import re
import pandas as pd
import diff_match_patch as dmp_module
from nltk import word_tokenize
from src.consts import *
from src.inference.factory import Factory
from src.inference.utils import retry_wrapper
from src.laquer_methods.llm_method_prompts import *
from src.laquer_methods.utils import find_substring
from src.decompose_to_facts import fix_local_offset_to_doc_offset
from src.lexical_alignment.lexical_edit_distance_attribution import word_tokenize_with_spans


class LLMBasedAlignment:
    def __init__(self, task, args):
        self.task = task
        self.args = args
        self.dmp = dmp_module.diff_match_patch()
        self.dmp.Match_Distance = 999999999999999999999  # cancel the loc parameter, we don't know where the llm would output a location
    
    def build_example(self, example, include_solution: bool = False):
        prompt = ""
        prompt += "\nInput: \n"
        source_spans = list(example['source_spans'].values())
        for source_idx, source_span in enumerate(source_spans):
            prompt += "Source " + str(source_idx + 1) + ": " + source_span + "\n"
        
        prompt += "\nOutput: " + example['sentence'] + "\n"
        
        prompt += "\nAttribution: \n"
        if include_solution:
            prompt += " ; ".join(example['spans_aligned'])
        
        return prompt
    
    def update_few_shot_granularity_based_on_input(self, topic, source_spans, source_granularity, documents):
        all_sources = {}
        if source_granularity == 'document':
            unique_doc_ids = set(['_'.join(source_key.replace(f"{topic}_", '').split('_')[:-1]) for source_key in source_spans.keys()])
            for doc_id in unique_doc_ids:
                all_sources[doc_id] = documents[doc_id]
        elif source_granularity == 'sentence':
            all_sources = source_spans
        else:
            raise ValueError(f"Granularity {source_granularity} not recognized")
                
        return all_sources

    def build_prompt(self, datapoint):
        prompt = ""
        
        prompt += ins_prompt + "\n\n"
        
        if self.task == MDS_TASK:
            few_shot = mds_few_shot
            few_shot_documents = mds_few_shot_documents
        elif self.task == LFQA_TASK:
            few_shot = lfqa_few_shot
            few_shot_documents = lfqa_few_shot_documents
        else:
            raise ValueError(f'unexpected task {self.task}')
        
        for few_shot_example in few_shot:
            few_shot_example = few_shot_example.copy()
            few_shot_example['source_spans'] = self.update_few_shot_granularity_based_on_input(few_shot_example['topic'], few_shot_example['source_spans'], datapoint['source_granularity'], few_shot_documents[few_shot_example['topic']])
            prompt += self.build_example(few_shot_example, include_solution=True) + "\n"
        prompt += self.build_example(datapoint, include_solution=False)

        return prompt

    def find_substring_fuzzy(self, text1, text2):
        result = self.dmp.match_main(text1, text2, loc=0)
        if result != -1:
            # heuristics for finding end token
            text2_words = word_tokenize(text2)
            
            # search the last word of text2 in text1
            try:
                last_idx_found = text1[result:].index(text2_words[-1])
                last_idx_found += len(text2_words[-1])
            except:
                # take the same number of words since
                text1_words = word_tokenize_with_spans(text1[result:])
                
                last_idx_found = text1_words[:len(text2_words)][-1][1][1]
                
            last_idx_found += result
            return (result, last_idx_found)

        return (-1, -1)
        


    def build_full_context_sources_from_metadata(self, datapoint):
        source_metadata = datapoint.get('source_metadata', {})
        grouped_sentences = {}
        for metadata_rows in source_metadata.values():
            for row in metadata_rows:
                document_id = row.get('documentFile')
                sent_text = row.get('docSentText')
                sent_idx = row.get('docSentCharIdx')
                if document_id is None or sent_text is None:
                    continue
                grouped_sentences.setdefault(document_id, []).append((int(sent_idx), sent_text))

        full_context_sources = {}
        for document_id, sentence_rows in grouped_sentences.items():
            seen_sentences = set()
            ordered_sentences = []
            for sent_idx, sent_text in sorted(sentence_rows, key=lambda x: x[0]):
                sentence_key = (sent_idx, sent_text)
                if sentence_key in seen_sentences:
                    continue
                seen_sentences.add(sentence_key)
                ordered_sentences.append(sent_text)
            if ordered_sentences:
                full_context_sources[document_id] = " ".join(ordered_sentences)

        if len(full_context_sources) > 0:
            return full_context_sources

        source_spans = datapoint.get('source_spans', {})
        for source_id, source_text in source_spans.items():
            document_id = source_id.split('__')[0]
            full_context_sources.setdefault(document_id, [])
            if source_text not in full_context_sources[document_id]:
                full_context_sources[document_id].append(source_text)

        return {document_id: " ".join(texts) for document_id, texts in full_context_sources.items() if len(texts) > 0}

    def parse_response(self, datapoint, response):
        primary_sources = datapoint['source_spans']

        def try_find_offset(output_span_alignment):
            for source_id, source_text in primary_sources.items():
                offset = find_substring(source_text, output_span_alignment.strip())
                if offset[0] != -1:
                    return source_id, source_text, offset, 'primary'

            for source_id, source_text in primary_sources.items():
                offset = self.find_substring_fuzzy(source_text, output_span_alignment.strip())
                if offset[0] != -1:
                    return source_id, source_text, offset, 'primary'

            return None, None, (-1, -1), None

        found_alignments = []
        try:
            response['text'].split(';')
        except:
            raise
        for output_span_alignment in response['text'].split(';'):
            if output_span_alignment.strip() == '':
                continue
            source_id, source_text, offset, alignment_source = try_find_offset(output_span_alignment)

            if offset[0] != -1 and offset[1] != -1:
                found_alignments.append({
                    "offset": offset,
                    "source_id": source_id,
                    "source_text": source_text,
                    "alignment_source": alignment_source,
                })
            else:
                raise ValueError(f"couldn't find text {output_span_alignment}")
        
        if len(found_alignments) == 0:
            raise ValueError(f"No output span found in source spans ; response['text']: {response['text']} ; sentence: {datapoint['sentence']}")
        
        alignments = []
        for found_alignment in found_alignments:
            offset = found_alignment['offset']
            source_id = found_alignment['source_id']
            source_text = found_alignment['source_text']
            source_span = source_text[offset[0]:offset[1]]
            if offset[1] == 0:
                raise ValueError("Unexpected offset 1 is 0 , saw this happening for LFQA for '5157963cac6c4e0de1e8f729ce29e23e706c13c82425038512b7f5526eab320d-neeva'")
            
            doc_id = source_id.split('__')[0]
            
            doc_sent_char_idx = None
            doc_sent_text = None
            if datapoint['source_granularity'] == 'sentence' and source_id in datapoint['source_metadata']:
                any_source_metadata = datapoint['source_metadata'][source_id][0]
                doc_sent_char_idx = int(any_source_metadata['docSentCharIdx'])
                doc_sent_text = any_source_metadata['docSentText']
                all_sources_offsets = [offset for source_metadata in datapoint['source_metadata'][source_id] for offset in source_metadata['docSpanOffsets']]
                offset = fix_local_offset_to_doc_offset(offset, all_sources_offsets)
            else:
                offset = [offset]
            
            alignments.append({
                "topic": datapoint['topic'],
                "fact_idx": datapoint['fact_idx'],
                "documentFile": doc_id,
                "docSpanOffsets": offset,
                "docSpanText": source_span,
                'docSentCharIdx': doc_sent_char_idx,
                'docSentText': doc_sent_text,
                "scuSentence": datapoint['sentence'],
            })
        
        return pd.DataFrame(alignments)
                

    def extract_attribution(self, datapoint):
        factory = Factory(self.args)

        inference_wrapper = factory.inference_wrapper()
        prompt = self.build_prompt(datapoint)

        try:
            return self._extract_attribution_w_retry(datapoint, prompt, inference_wrapper)
        except Exception as e:
            # Revert to original attribution
            results = pd.DataFrame([source_row for source_rows in datapoint['source_metadata'].values() for source_row in source_rows])
            response = {
                'error': str(e)
            }
            datapoint = datapoint.copy()
            datapoint.pop('source_metadata')
            return {
                "results": results,
                **response,
                **datapoint
            }

    @retry_wrapper
    def _extract_attribution_w_retry(self, datapoint, prompt, inference_wrapper):
        response = inference_wrapper.generate_text(messages=[{"role": "user", "content": prompt}])

        try:
            results = self.parse_response(datapoint, response)
            final_response = response
        except ValueError as e:
            has_required_alignment_fields = isinstance(datapoint.get('source_spans'), dict) and isinstance(datapoint.get('source_metadata'), dict)
            full_context_sources = self.build_full_context_sources_from_metadata(datapoint) if has_required_alignment_fields else {}
            has_full_context = len(full_context_sources) > 0
            if not has_full_context:
                raise

            full_context_datapoint = datapoint.copy()
            full_context_datapoint['source_spans'] = full_context_sources
            full_context_datapoint['source_granularity'] = 'document'
            full_context_prompt = self.build_prompt(full_context_datapoint)
            full_context_response = inference_wrapper.generate_text(messages=[{"role": "user", "content": full_context_prompt}])

            results = self.parse_response(full_context_datapoint, full_context_response)
            final_response = {
                **full_context_response,
                'window_parse_error': str(e),
                'used_full_context_retry': True,
            }

        datapoint = datapoint.copy()
        datapoint.pop('source_metadata')
        return {
                "results": results,
                **final_response,
                **datapoint
            }
