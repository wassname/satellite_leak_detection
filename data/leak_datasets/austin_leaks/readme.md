
Data received from Austin Water as part of public information request: PIR 32348 on February 16, 2017. Locations are the nearest street address and no other contextual information was given.

Parsing the data, working out the columns and other details are in `notebooks/0_process_data/process_austin_leaks.ipynb`.

----------

Dates are in US/Central timezone.

Locations seem to be EPSG:6578.

Columns:

The columns are unlabelled but the top left 47486 match leaks pending inspections https://services1.arcgis.com/PuB3FWUAxkScvfQy/ArcGIS/rest/services/LPI/FeatureServer/0

The bottom right data matches leak pending repair https://services1.arcgis.com/PuB3FWUAxkScvfQy/ArcGIS/rest/services/LWOPR/FeatureServer/0

```py
columns = [
 'OBJECTID',
 'INSPFLAG',
 'PRI',
 'PROB',
 'PROBDTTM', # Problem datetime US/Central timezone.
 'SCHEDDTTM', # Scheduled datetime US/Central timezone.
 'SERVNO',
 'MAPNO',
 'STARTDTTM', # Start datetime US/Central timezone.
 'PROBCODE',
 'PROBDESC',
 'CITY',
 'PREDIR',
 'STNAME',
 'STNO',
 'STSUB',
 'SUFFIX',
 'ZIP',
 'ADDRKEY',
 'FullStreetName',
 'X', # X location in EPSG:6578
 'Y', # Y location in EPSG:6578
 '22', # CMPLKEY (Complementary key?, INT)?, SERVER (nah thats 47486-627096, Objectid (no too high)?? 1913-1496485.
 'QTYCALLS', # Quantity of calls?
 'INITDTTM', # Initial date time US/Central timezone.
 'WONO', # Work order number?
 'LOC',
 'DESCRIPT', # Description
 'COMPDTTM' # Completion data time US/Central timezone.
]
```
