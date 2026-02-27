import sqlite3
import faiss
import os

db_path = '../data/db.sqlite'
index_path = '../data/index.faiss'

with open('diag_output.txt', 'w') as out:
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            # Images
            c.execute('SELECT count(*) FROM images')
            img_count = c.fetchone()[0]
            out.write(f'IMAGE_COUNT: {img_count}\n')
            
            # People
            c.execute('SELECT count(*) FROM people')
            people_count = c.fetchone()[0]
            out.write(f'PEOPLE_COUNT: {people_count}\n')
            
            # Albums
            c.execute('SELECT count(*) FROM albums')
            albums_count = c.fetchone()[0]
            out.write(f'ALBUMS_COUNT: {albums_count}\n')
            
            # Faces
            c.execute('SELECT count(*) FROM faces')
            faces_count = c.fetchone()[0]
            out.write(f'FACES_COUNT: {faces_count}\n')

            conn.close()
        else:
            out.write(f'DB_NOT_FOUND: {db_path}\n')

        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            out.write(f'INDEX_EXISTS: True\n')
            out.write(f'FAISS_COUNT: {index.ntotal}\n')
        else:
            out.write(f'INDEX_NOT_FOUND: {index_path}\n')
            
    except Exception as e:
        out.write(f'ERROR: {str(e)}\n')
