import sqlite3, os
path=os.path.join('data','db.sqlite')
print('db path', path, os.path.exists(path))
conn=sqlite3.connect(path)
cur=conn.cursor()
cur.execute('PRAGMA table_info(images)')
cols=cur.fetchall()
print('columns:')
for col in cols:
    print(col)
conn.close()
