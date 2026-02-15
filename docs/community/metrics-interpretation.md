---
layout: "default"
title: "Metrics Interpretation"
nav_order: 8
parent: "Smart Control Project Documentation"
---

# Understanding the Logs in the Metrics directory

The metrics files (observation response, action response, reward infos, zone info, and device info) are saved in the `metrics` path specified by you in the SAC_Demo notebook, organized by directories corresponding to different runs and each metric file corresponding to the metric for a single day. These can be read using the functions in smart_control.controller_reader.py, which starts by creating a ProtoReader with the path to the metrics directory. This section will explain the components of the different metric logs. 

## Observation Responses

For every five minutes, i.e. 300 second differences between timestamps, this log first lists all observation requests, i.e. which measurements for which sensor (identified by the device id).
Then, it lists each observation response, which contains the timestamp, value for the measurement, and the validity of the measurement (true or false), as well as the corresponding request. 
You can read the log file using ProtoReader.read_observation_responses(start_timestamp, end_timestamp). 
Ex. reader = controller_reader.ProtoReader('dataset/SB1/19/')
reader.read_observation_responses(pd.Timestamp('2019-03-16T00:00:00'), pd.Timestamp('2019-03-17T00:00:00'))
Each timestamp in the result is of type protobuf.Timestamp, which can be converted to an interpretable pd.Timestamp object using the protos_to_pandas_timestamp() function in smart_control.utils. 

## Action Responses
Requests entail supply air temperature at two different devices and supply water temperature at one device, corresponding to action values.  Responses detail how the given device is responding to requested new value.

## Reward Responses
Split up into 5 minute intervals again, this provides the breakdown of the reward value at that time step. 

## Reward Infos 
These are organized in 5 minute intervals, with each entry organized by zone ID corresponding to the zone info file. Each zone contains the measurements providing the values for the described metrics corresponding to the agent_id (ex. "baseline_policy") and scenario_id (ex. "baseline_collect").  

## Device Info
This provides information regarding each device, identified by a unique device ID and code as well as its namespace and type. Each device has a specific set of observable fields and action fields that are listed underneath its identifiers. 

## Zone Info
Each entry represents a specific zone identified by zone_id and zone_descriotion, inside a building identified by building_id on the given floor. It also contains a list of IDs for the devices present inside the given zone. 

[Back to Home](../index.md)