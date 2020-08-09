# -*- coding: utf-8 -*-
from model.xgb import get_xgb_prediction
from consumer import Kafka_consumer
from producer import Kafka_producer


# res = get_xgb_prediction(test_season=['20180930'], load=True)


def handle_request():
    KAFAKA_HOST = "47.103.137.116"
    KAFAKA_PORT = 9092
    KAFKA_TOPIC_REC = 'topic_rec1'
    KAFKA_TOPIC_SEND = 'topic_send1'

    consumer = Kafka_consumer(KAFAKA_HOST, KAFAKA_PORT, KAFKA_TOPIC_REC)
    producer = Kafka_producer(KAFAKA_HOST, KAFAKA_PORT,KAFKA_TOPIC_SEND, key='predictions')

    print("===========> consumer:", consumer)
    message = consumer.consume_data()
    for msg in message:

        print('msg---------------->k,v', msg.key, msg.value)

        if msg.key == b'endDate':
            print("===========> producer:", producer)
            msg.value.decode('utf-8')
            print(msg.value.decode('utf-8'))
            try:
                res = get_xgb_prediction(test_season=[msg.value.decode('utf-8')], load=True)
                res_columns = res.columns
                res = res.iloc[:100]
                predictions = []
                for i, (_, row) in enumerate(res.iterrows()):
                    predictions.append({k: row[k] for k in res_columns})
                params = {'stockDataDetail':predictions, 'predictStock':['aaa'], 'realStock':['bbb']}
                producer.sendjsondata(params)

            except Exception as e:
                params = {'stockDataDetail':[e], 'predictStock':['aaa'], 'realStock':['bbb']}
                producer.sendjsondata(params)
        # else:
            # producer.sendstrdata('wrong key')
            # print('wrong key')

if __name__ == '__main__':
    handle_request()

