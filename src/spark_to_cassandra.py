from cassandra.cluster import Cluster
from ssl import SSLContext, PROTOCOL_TLSv1_2 , CERT_REQUIRED
from cassandra.auth import PlainTextAuthProvider
from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement,BatchStatement,BatchType

from .cassandra_credentials import get_creds

CREDS = get_creds()

ssl_context = SSLContext(PROTOCOL_TLSv1_2 )
ssl_context.load_verify_locations('sf-class2-root.crt')
ssl_context.verify_mode = CERT_REQUIRED
auth_provider = PlainTextAuthProvider(username=CREDS["USERNAME"],
                                      password=CREDS["PASSWORD"])
cluster = Cluster([CREDS["CLUSTER_LINK"]], ssl_context=ssl_context, auth_provider=auth_provider, port=9142)
session = cluster.connect()
#Select Statement
r = session.execute('SELECT * FROM testing_node.predictions')
print(r.current_rows)

count = 1
for i in range(0,2):
  insert_preds = session.prepare('INSERT INTO testing_node.predictions (vals,prediction) VALUES (?,?)')
  batch = BatchStatement(consistency_level=ConsistencyLevel.LOCAL_QUORUM,batch_type=BatchType.UNLOGGED)
  temp = dbc.loc[0:10,'prediction'].values.tolist()
  for j in temp:
    batch.add(insert_preds,[count, int(j)])
    count += 1
  session.execute(batch)
  r = session.execute('SELECT * FROM testing_node.predictions')
  print(r.current_rows)
