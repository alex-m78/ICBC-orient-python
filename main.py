# -*- coding: utf-8 -*-
from model.xgb import get_xgb_prediction
from consumer import Kafka_consumer
from producer import Kafka_producer


# res = get_xgb_prediction(test_season=['20180930'], load=True)


def handle_request():
    KAFAKA_HOST = "47.103.137.116"
    KAFAKA_PORT = 9092
    KAFKA_TOPIC = 'topic002'

    consumer = Kafka_consumer(KAFAKA_HOST, KAFAKA_PORT, kafkatopic=KAFKA_TOPIC)
    producer = Kafka_producer(KAFAKA_HOST, KAFAKA_PORT,
                              kafkatopic=KAFKA_TOPIC, key='predictions')

    print("===========> consumer:", consumer)

    message = consumer.consume_data()
    for msg in message:

        print('msg---------------->k,v', msg.key, msg.value)

        if msg.key == b'end_date':
            print("===========> producer:", producer)
            res = get_xgb_prediction(test_season=[msg.value], load=True)
            res_columns = res.columns
            for i, (_, row) in enumerate(res.iterrows()):
                params = [{k: row[k] for k in res_columns}]
                producer.sendjsondata(params)


if __name__ == '__main__':
    handle_request()
