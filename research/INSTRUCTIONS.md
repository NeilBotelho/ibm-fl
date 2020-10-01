1. Download the cancer data into the research directory in a folder called source_data
its structure should be as follows:
```
research
|
----source_data
		|
		-----train
		|		|
		|		-----malignant
		|		|       0.jpg.....
		|		-----benign
		|				10.jpg
		|
		-------test
				same as above
```


2. With current directory as top of the ibmfl repo, run:
	
		python research/generate_configs.py  -n 2 -d l -p research/data/te2_50/ -m keras

	to create data for each party. You can customise number of parties and number of data points per party as well. Use 
		
		python  examples/generate_configs.py -h
	
	for a list of all customisation options
	
3. Similarly run 
 
 	Example Command for 2 parties, with 200 and 150 data points resp with 0.25 and 0.75 B:M ratio.
 
	for unbalanced non IID,
	
		python research/generate_data.py  -n 2 -pp 200 150 -r 0.25 0.75 --name 4p_200pp
	
	for balanced non IID,
	
		python research/generate_data.py  -n 2 -pp 200 -r 0.25 0.75 --name 4p_200pp
	
	and for IID with default ratios,
	
		python research/generate_data.py  -n 2 -pp 200 --name 4p_200pp
	
	to specify the same ratio for all parties,
	
		python research/generate_data.py  -n 2 -pp 200 -r 0.25 --name 4p_200pp

	to generate the required configuration files

4. From here you can basically follow the quick start guide in the root of the repository but substitute ```examples``` with ```research```
