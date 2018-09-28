from torchtext.datasets import TranslationDataset
import os

from ..common.downloadable import Downloadable

from torchtext.data import Field


class EuroParl(Downloadable, TranslationDataset):
    def __init__(self, root: str, language: str, en_field: Field, lang_field: Field,
                 from_english: bool = False, download=False):

        Downloadable.__init__(self, root, EuroParl.url(language), download)
        
        self.language = language
        self.from_language = "en" if from_english else language
        self.to_language = "en" if not from_english else language
        
        self.setup()
        
        TranslationDataset.__init__(
            self,
            path=os.path.join(root, f"europarl-v7.{language}-en."),
            exts=("en", language) if from_english else (language, "en"),
            fields=(en_field, lang_field) if from_english else (lang_field, en_field)
        )
        
    def exists(self):
        prefix = os.path.join(self.root, f"europarl-v7.{self.language}-en.")
        
        return os.path.exists(prefix + "en") and os.path.exists(prefix + self.language) 
        
    @staticmethod
    def url(language: str):
        return f"http://www.statmt.org/europarl/v7/{language}-en.tgz"