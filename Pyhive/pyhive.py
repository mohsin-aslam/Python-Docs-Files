from pyhive import hive
cursor = hive.connect('10.1.1.184:10000').cursor()
cursor.execute('''INSERT into table testtablenew  VALUES(5,'Mohsin22')''', async=True)