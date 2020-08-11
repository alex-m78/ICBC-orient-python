# -*- coding: utf-8 -*-
from model.xgb import get_xgb_prediction
from consumer import Kafka_consumer
from producer import Kafka_producer

availble_season = ['20160331', '20160630', '20160930', '20170331', '20170630', '20170930',
                   '20180331', '20180630', '20180930', '20190331', '20190630', '20190930', '20200331']

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

        if msg.key == b'endDate' and msg.value.decode('utf-8') in availble_season:
            print("===========> producer:", producer)
            # msg.value.decode('utf-8')
            print(msg.value.decode('utf-8'))

            res, predicted_and_real, acc, p30, count_predicted, count_real = get_xgb_prediction(test_season=[msg.value.decode('utf-8')], load=True, read_sql=False)
            res_columns = res.columns
            res = res.iloc[:100]
            predictions,predicted, real = [], [], []
            for i, (_, row) in enumerate(res.iterrows()):
                predictions.append({k: row[k] for k in res_columns})
            for i, (_, row) in enumerate(predicted_and_real.iterrows()):
                predicted.append({k: row[k] for k in ['ts_code_predicted','name_predicted','label_new']})
                real.append({k: row[k] for k in ['ts_code_real','name_real']})
            params = {'stockDataDetail':predictions, 'predictStock':predicted, 'realStock':real, 'accuracy':acc,
                      'precisionTop30':p30, 'countPredict':{'label':list(count_predicted.keys()),'count':list(count_predicted.values())},
                      'countReal':{'label':list(count_real.keys()),'count':list(count_real.values())}}
            producer.sendjsondata(params)

            #except Exception as e:
                #params = {'stockDataDetail':[e], 'predictStock':['aaa'], 'realStock':['bbb']}
                #producer.sendjsondata(params)
                #print(e)
        # else:
            # producer.sendstrdata('wrong key')
            # print('wrong key')

if __name__ == '__main__':
    handle_request()

