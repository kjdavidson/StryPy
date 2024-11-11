# StryPy

Used for reading data from Stryde nodes into ObsPy environment. Tested against v1.03 (1 October 2022) of the Stryde implementation of SEG-D revision 3.0.

## Installation:

Tested with Python 3.11 and ObsPy 1.4.0

```
pip install -e .
```

## Usage:

In a Python script:
```
from strypy import read_stryde
stream = read_stryde('file.segd')
```

Full headers:
```
general_header, scan_header, json_header, trace_header, stream = read_stryde('file.segd', stryde_headers=True)
```

To trim stream to UTCDateTime definitions:
```
from obspy import UTCDateTime
starttime = UTCDateTime("2022-12-23T13:00:00")
endtime = UTCDateTime("2022-12-23T13:30:00")
stream = read_stryde('file.segd', starttime=starttime, endtime=endtime)
```

From the command line (convert to SAC or MINISEED)
```
python3 strypy -o SAC file.segd
python3 strypy -o MSEED file.segd
```
