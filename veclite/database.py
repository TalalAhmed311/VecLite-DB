import sqlite3
import os
class SqliteDB:


    def __init__(self,path='vector_store',db_name='vector_database') -> None:

        # Connect to the SQLite database (creates a new file if it doesn't exist)
        if not os.path.exists(os.path.join(os.getcwd(),path)):
            os.mkdir(os.path.join(os.getcwd(),path))

        db = os.path.join(os.getcwd(),path,db_name)
        self.con = sqlite3.connect(f'{db}.db')
        self.cursor = self.con.cursor()

        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS centroids (
                centroid_id VARCHAR PRIMARY KEY,
                centroid_vector BLOB NOT NULL
            )
        ''')


        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vector_db (
                id VARCHAR PRIMARY KEY,
                centroid_id VARCHAR,  
                my_vector BLOB NOT NULL,
                metadata TEXT,
                content TEXT,
                FOREIGN KEY (centroid_id) REFERENCES centroids(centroid_id)
            )
        ''')


    def add_centroids(self,centroids_data_list):
        
     try:

        query = 'INSERT INTO centroids (centroid_id, centroid_vector) VALUES (?, ?)'

        self.cursor.executemany(query, centroids_data_list)
        self.con.commit()
        
     except Exception as e:
         print(f"Error adding vectors: {e}")


    def add_vectors(self,vector_model_data):
        try:

            query = "INSERT INTO vector_db (id, centroid_id, my_vector, metadata, content) VALUES (?, ?, ?, ?, ?)"

            self.cursor.executemany(query, vector_model_data)

            self.con.commit()

            print("Vectors added successfully.")
        except Exception as e:
            print(f"Error adding vectors: {e}")



    def delete_all(self):
        try:

            self.cursor.execute("DELETE FROM centroids")
            self.cursor.execute("DELETE FROM vector_db")

            self.con.commit()

            print("Succesfully deleted")
        except Exception as e:
            print(f"Error while deleting {e}")

    def fetch_centroids(self):
        try:

            self.cursor.execute("SELECT * FROM centroids")
            result = self.cursor.fetchall()

            return result
        except Exception as e:
            print(f"Error adding vectors: {e}")


    def fetch_cluster_vectors(self,centroid_id):
        try:

            query = '''
                SELECT * FROM vector_db
                WHERE centroid_id = ?
            '''

            self.cursor.execute(query, (centroid_id,))
            result = self.cursor.fetchall()

            return result
        except Exception as e:
            print(f"Error adding vectors: {e}")

    def fetch_vectors(self):

        try:

            self.cursor.execute("SELECT * FROM vector_db")
            result = self.cursor.fetchall()

            return result
        except Exception as e:
            print(f"Error adding vectors: {e}")