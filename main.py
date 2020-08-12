# -*- coding: utf-8 -*-
from model.xgb import get_xgb_prediction
from consumer import Kafka_consumer
from producer import Kafka_producer

availble_season = ['20160331', '20160630', '20160930', '20170331', '20170630', '20170930',
                   '20180331', '20180630', '20180930', '20190331', '20190630', '20190930', '20200331']

def handle_request():
    KAFAKA_HOST = "47.103.137.116"
    KAFAKA_PORT = 9092
    KAFKA_TOPIC_REC = 'topic_rec2'
    KAFKA_TOPIC_SEND = 'topic_send2'

    consumer = Kafka_consumer(KAFAKA_HOST, KAFAKA_PORT, KAFKA_TOPIC_REC,group_id='test1')
    producer = Kafka_producer(KAFAKA_HOST, KAFAKA_PORT,KAFKA_TOPIC_SEND, key='predictions')

    print("===========> consumer:", consumer)
    message = consumer.consume_data()
    for msg in message:
        print('msg---------------->k,v', msg.key, msg.value)

        if msg.key == b'endDate':
            print("===========> producer:", producer)
            # msg.value.decode('utf-8')
            print([msg.value.decode('utf-8')])

            res, predicted_and_real, acc, p30, count_predicted, count_real = get_xgb_prediction(
                test_season=['20200331'], load=True, read_sql=False)
            res_columns = res.columns
            res = res.iloc[:100]

            predictions, predicted, real = [], [], []
            industryDataPre, industryDataReal = [], []

            for i, (_, row) in enumerate(res.iterrows()):
                predictions.append({k: row[k] for k in res_columns})
            for i, (_, row) in enumerate(predicted_and_real.iterrows()):
                predicted.append({k: row[k] for k in ['ts_code_predicted', 'name_predicted', 'label_new']})
                real.append({k: row[k] for k in ['ts_code_real', 'name_real']})
            for k, v in count_predicted.items():
                industryDataPre.append({'industryName': k, 'count': v})
            for k, v in count_real.items():
                industryDataReal.append({'industryName': k, 'count': v})

            params = {'stockDataDetail': predictions, 'predictStock': predicted, 'realStock': real,
                      'accuracy': round(acc, 3),
                      'precisionTop30': round(p30, 3), 'industryDataPre': industryDataPre,
                      'industryDataReal': industryDataReal}
            producer.sendjsondata(params)


if __name__ == '__main__':

    handle_request()