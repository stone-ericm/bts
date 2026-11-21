"""Use the permitted filename spelling of the KILL case after a fixture path typo."""
import probe_text


probe_text.HOSTILE = tuple(name.replace("status=9/KILL", "signal=kill") for name in probe_text.HOSTILE)
probe_text.main()
