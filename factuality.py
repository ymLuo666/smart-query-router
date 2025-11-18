# Adapted from https://github.com/BunsenFeng/Knowledge_Card/blob/main/factuality.py

import transformers
import numpy as np
import wikipedia as wp
import time

class FactualChecker:
    def __init__(self):
        self._load_factuality_model([-1, -1]) # run on CPU

    def _load_factuality_model(self, devices):
        tokenizer1 = transformers.AutoTokenizer.from_pretrained("tals/albert-xlarge-vitaminc-mnli")
        tokenizer2 = transformers.AutoTokenizer.from_pretrained("roberta-base")
        self.vitaminc_model = transformers.pipeline('text-classification', model="tals/albert-xlarge-vitaminc-mnli", tokenizer = tokenizer1, device = devices[0], return_all_scores = True, max_length = 512, truncation = True)
        self.factkb_model = transformers.pipeline('text-classification', model="bunsenfeng/FactKB", tokenizer = tokenizer2, device = devices[1], return_all_scores = True, max_length = 512, truncation = True)

    def factuality(self, text):
        docs = [""]
        try:
            for ent in wp.search(text[:100], results = 3):
                try:
                    docs.append(wp.summary(ent, sentences=5))
                except:
                    # print("error in retrieving summary for " + ent)
                    pass
        except:
            print("error in wiki search")
            time.sleep(2)
            pass
        #docs = [wp.summary(ent, sentences=10) for ent in wp.search(text, results = 5)]

        if self.vitaminc_model is None or self.factkb_model is None:
            raise Exception("factuality model not loaded")
        scores = []
        # for doc in docs:
        #     text_post = text + " " + doc
        #     vitaminc_score = vitaminc_model(text_post)
        #     factkb_score = factkb_model(text_post)
        #     print(vitaminc_score)
        #     print(factkb_score)
        #     exit(0)
        text_posts = [text + " " + doc for doc in docs]
        vitaminc_scores = self.vitaminc_model(text_posts)
        factkb_scores = self.factkb_model(text_posts)
        for i in range(len(docs)):
            vitaminc_score = (vitaminc_scores[i][0]['score'] - vitaminc_scores[i][1]['score'] + 0 * vitaminc_scores[i][2]['score'] + 1) / 2 # 0 to 1
            factkb_score = factkb_scores[i][1]['score'] # 0 to 1
            scores.append((vitaminc_score + factkb_score) / 2)
        return np.max(scores)

    def factuality_check(self, text: str, threshold=0.5) -> bool:
        '''Return True if factuality score exceed threshold; Otherwise False.'''
        if self.vitaminc_model is None or self.factkb_model is None:
            raise Exception("factuality model not loaded")
      
        try:
            score = self.factuality(text) > threshold 
            return score >= threshold
        except:
            print(f'cannot handle {text}')
                
# test
if __name__ == "__main__":
    checker = FactualChecker()
    texts = ["The capital of the United States is Washington, D.C.", "I am stupid", "I am a genius", "i am a", "a a a", "b b b", "c c c", "d d d"]
    for text in texts:
        print(f'{checker.factuality_check(text)}: {text}')
