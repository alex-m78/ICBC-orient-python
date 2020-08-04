# -*- coding: utf-8 -*-

import sys
import time
import json
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError
from model.xgb import get_xgb_prediction


class Kafka_producer():

    def __init__(self, kafkahost, kafkaport, kafkatopic, key):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.key = key
        print("producer:h,p,t,k", kafkahost, kafkaport, kafkatopic, key)
        bootstrap_servers = '{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort
        )
        print("boot svr:", bootstrap_servers)
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers
                                      )

    def sendjsondata(self, params):
        try:
            parmas_message = json.dumps(params, ensure_ascii=False)
            producer = self.producer
            v = parmas_message.encode('utf-8')
            k = self.key.encode('utf-8')
            print("send msg:(k,v)", k, v)
            producer.send(self.kafkatopic, key=k, value=v)
            producer.flush()
        except KafkaError as e:
            print(e)


def main(end_date='20180630'):
    key = 'end_date'
    KAFAKA_HOST = "localhost"
    KAFAKA_PORT = 9092
    KAFAKA_TOPIC = "test"
    producer = Kafka_producer(KAFAKA_HOST, KAFAKA_PORT, KAFAKA_TOPIC, key)
    print("===========> producer:", producer)
    producer.sendjsondata([{'end_date': end_date}])


if __name__ == '__main__':
    main('20180930')
