from ..utils import file as utils


class Downloadable:
    def __init__(self, root: str, src: str, download: bool = False):
        self.root = root
        self.url = src
        self.do_download = download
                
    def download(self):
        if self.exists():
            print("Already downloaded!")
        else:
            utils.download_extract_try(self.url, self.root)
            self.download_finished()
        
    def download_finished(self):
        return
        
    def setup(self):
        if self.do_download:
            self.download()
        
        if not self.exists():
            print("Setup failed: not downloaded!")
        else:
            self._setup()
            
    def _setup(self):
        return
        
    def exists(self):
        return
