# Description
Movement detection application based on background subtraction.

# Configuration
Adjust the `cont_area_thr` in interval 0-1 in the conig file to adjust detection sensitivity.  
 
## Usage
```
usage: run_detection.py [-h] config

Run the lightweight detection algorithm

positional arguments:
  config             path to the configuration file

optional arguments:
  -h, --help         show this help message and exit
```

Run the movement detection:
```
python3 run_detection.py path_to_config.yml
```  


## Related
1. [Bundle into a single executable](doc/pyinstaller.md)
2. [Streaming server](doc/streaming_server.md)
