import mysql.connector

from Security.Encryption import AESCipher
from src.config.CommonVariables import RegisterUser, property_var, get_settings
# from Security.OAuth2Security import RegisterUser
from src.config.ExtractProperty import Property

props = property_var.get_property_data()
encrypter = AESCipher(get_settings().ENCODING_SALT)

class MysqlDB:

    def __init__(self):
        self.cnx = mysql.connector.connect(
            host=props["mysql"]["host"],
            port=props["mysql"]["port"],
            user=encrypter.decrypt(props["mysql"]["user"]),
            password=encrypter.decrypt(props["mysql"]["password"]),
            database=props["mysql"]["database"]
        )
        self.cur = self.cnx.cursor()

    def reestablish_connection(self):
        print("Called MysqlDB.start_connection()")
        self.cnx.reconnect()

    def enrich_user_result(self, columns, result_array):
        result = []
        for row in result_array:
            row_dict = dict()
            for i in range(len(columns)):
                row_dict[columns[i]] = row[i]
            result.append(row_dict)
        return result

    def get_user_by_username(self, username):
        print("Calling MysqlDB.get_user_by_username()")
        if self.cnx.is_connected():
            print("MySQL Connection is active")
        else:
            # self.start_connection()
            self.reestablish_connection()
            print("MySQL Connection is not active")
        self.cur.execute("SELECT * FROM yt_comm_user where username = %s", (username, ))
        desc = self.cur.description
        columns = [col[0] for col in desc]
        row = self.cur.fetchall()
        result = self.enrich_user_result(columns=columns, result_array=row)
        return result

    def create_user(self, user: RegisterUser):
        print("Calling MysqlDB.create_user()")
        print(user)
        if self.cnx.is_connected():
            print("MySQL Connection is active")
        else:
            self.start_connection()
            print("MySQL Connection is not active")
        sql_insert_query = "INSERT INTO yt_comm_user (username, full_name, email, hashed_password, disabled) VALUES (%s, %s, %s, %s, %s)"
        self.cur.execute(sql_insert_query, (user.username, user.full_name, user.email, user.hashed_password, user.disabled))
        self.cnx.commit()

    def close_connection(self):
        print("Called MysqlDB.close_connection()")
        self.cnx.close()


if __name__ == "__main__":
    property = Property()
    props = property.get_property_data()
    print(props)
    # mysqlDB = MysqlDB()
    # mysqlDB.start_connection()
    # res = mysqlDB.get_user_by_username("admin2")
    # print(res)
    # mysqlDB.close_connection()
    # desc = [('username', 253, None, None, None, None, 0, 20483, 255), ('full_name', 253, None, None, None, None, 1, 0, 255), ('email', 253, None, None, None, None, 1, 0, 255), ('hashed_password', 252, None, None, None, None, 1, 16, 255), ('disabled', 1, None, None, None, None, 1, 32768, 63)]
    # columns = [col[0] for col in desc]
    # row = [['admin', 'Admin User', 'admin@admin.com', '$2b$12$G6Qw5e.K871doase2mJqgepPaB7frMIWb973E9zspNl3dNrHSik8C', 0]]
    # print(mysqlDB.enrich_result(columns=columns, result_array=row))

