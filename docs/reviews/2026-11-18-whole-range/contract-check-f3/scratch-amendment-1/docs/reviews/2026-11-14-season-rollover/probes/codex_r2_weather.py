import sys,os,io,tempfile,shutil,json
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=['/Users/eric/projects/bts-pp-c1/src']
import bts.model.predict as model
import bts.data.pull as pull
source=Path(next(l.split(' ',1)[1] for l in Path('/tmp/g2-r2-cold.out').read_text().splitlines() if l.startswith('ROOT ')))
root=Path(tempfile.mkdtemp(prefix='g2-r2-weather-'));(root/'data/raw/2027').mkdir(parents=True);(root/'data/processed').mkdir();(root/'data/models').mkdir()
shutil.copyfile(source/'data/processed/pa_2026.parquet',root/'data/processed/pa_2026.parquet')
(root/'data/raw/2027/9001_weather.json').write_text('{"humidity":50}')
old=os.getcwd();os.chdir(root)
try:
    with patch.object(pull,'retry_urlopen',lambda *a,**k:io.BytesIO(b'{"dates":[]}')),patch.object(model,'urlopen',lambda *a,**k:io.BytesIO(b'{"dates":[]}')):
        try:model.run_pipeline('2027-03-24');result='success'
        except Exception as e:result=type(e).__name__+': '+str(e)
    print('WEATHER_ONLY_COLD_START',result,'root',root)
finally:os.chdir(old)
