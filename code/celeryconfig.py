from kombu import Queue
broker_url = ''
result_backend = 'rpc://'

task_serializer = 'pickle'
result_serializer = 'pickle'
accept_content = ['pickle']
timezone = 'Europe/Berlin'
enable_utc = True
task_default_queue = 'angular_task'
task_queues = (    
    Queue('angular_task', routing_key='angular.#'),
)