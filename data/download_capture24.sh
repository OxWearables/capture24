#!/bin/sh

# Create directory to contain all data
mkdir -p capture24
cd capture24/

# Download the 151 accelerometer data files
for i in $(seq -w 151)
do
    curl -JLO "https://ora.ox.ac.uk/objects/uuid:92650814-a209-4607-9fb5-921eab761c11/download_file?safe_filename=P${i}.csv.gz&type_of_work=Dataset"
done

# Download remaining files
curl -JLO "https://ora.ox.ac.uk/objects/uuid:92650814-a209-4607-9fb5-921eab761c11/download_file?safe_filename=metadata.csv&type_of_work=Dataset"
curl -JLO "https://ora.ox.ac.uk/objects/uuid:92650814-a209-4607-9fb5-921eab761c11/download_file?safe_filename=annotation-label-dictionary.csv&type_of_work=Dataset"

# Return to previous working directory
cd -
