"""Reuse the real-scorer corruption loop for the adopted module's unexercised JSON payloads."""
from pathlib import Path
H=Path(__file__).resolve().parent
source=(H/'r2_probe.py').read_text()
prefix=source.split('for label, day in CASES:')[0]
prefix=prefix.replace('r2-probe-{MODE}','r2-more-corrupt-{MODE}').replace('r2-fixtures-{MODE}','r2-more-corrupt-fixtures-{MODE}')
body=source[source.index('for label, payload in CORRUPT.items():'):source.index('# Literal spelling of the applied marker')]
payloads="CORRUPT = {'empty': b'', 'string': b'\"junk\"', 'bool': b'true', 'huge-int': b'{\"season\":2027,\"streak\":' + b'9'*5000 + b'}'}\n"
exec(compile(prefix+payloads+body+'\nOUT.close()\n',__file__,'exec'))
