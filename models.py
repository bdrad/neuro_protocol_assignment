from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin


class Preprocessor(TransformerMixin):
    def _process(self, report):
        report = report.lower()
        tokens = word_tokenize(report)

        to_remove = set(stopwords.words('english')) | set(",.;'()!?")
        tokens = filter(lambda t: t not in to_remove, tokens)

        return " ".join(tokens)

    def transform(self, X, *_):
        return list(map(self._process, X))
    
    def fit(self, *_):
        return self

def get_log_reg_model(data, labels):
    text_clf = Pipeline([
        ('preproc', Preprocessor()),
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(penalty='l1'))
    ])
    scores = cross_val_score(text_clf, data, labels, cv=10)
    text_clf.fit(data, labels)
    return text_clf, scores
