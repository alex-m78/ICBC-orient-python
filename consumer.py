from kafka import KafkaConsumer
from kafka.errors import KafkaError


class Kafka_consumer():

    def __init__(self, kafkahost, kafkaport, kafkatopic, group_id='test'):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.consumer = KafkaConsumer(
            self.kafkatopic,
            bootstrap_servers='{kafka_host}:{kafka_port}'.format(
                kafka_host=self.kafkaHost,
                kafka_port=self.kafkaPort),
            group_id=group_id
        )
        self.consumer.subscribe(topics=['topic002'])

    def consume_data(self):
        try:
            for message in self.consumer:
                yield message
        except KeyboardInterrupt as e:
            print(e)


def main():
    KAFAKA_HOST = "localhost"
    KAFAKA_PORT = 9092
    KAFAKA_TOPIC = "result"
    consumer = Kafka_consumer(KAFAKA_HOST, KAFAKA_PORT, KAFAKA_TOPIC)
    print("===========> consumer:", consumer)
    message = consumer.consume_data()
    for msg in message:
        if msg.key == 'predictions':
            print('msg---------------->v', msg.value)


if __name__ == '__main__':
    main()
