import yaml
import inspect
import os

def get_caller_file_name():
    call_stack = inspect.stack()
    call_filenames = [stack.filename for stack in call_stack]
    # call_filenames = ['C:\\Users\\spand\\Downloads\\PROJECTS\\Youtube Comment Analysis\\youtube-comment-analysis-app-python\\src\\config\\ExtractProperty.py', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap_external>', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap>', 'C:\\Users\\spand\\Downloads\\PROJECTS\\Youtube Comment Analysis\\youtube-comment-analysis-app-python\\src\\db\\MySQLDB.py', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap_external>', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap>', 'C:\\Users\\spand\\Downloads\\PROJECTS\\Youtube Comment Analysis\\youtube-comment-analysis-app-python\\Security\\OAuth2Security.py', 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2024.1.1\\plugins\\python-ce\\helpers\\pydev\\_pydev_imps\\_pydev_execfile.py', 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2024.1.1\\plugins\\python-ce\\helpers\\pydev\\pydevd.py', 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2024.1.1\\plugins\\python-ce\\helpers\\pydev\\pydevd.py', 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2024.1.1\\plugins\\python-ce\\helpers\\pydev\\pydevd.py', 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2024.1.1\\plugins\\python-ce\\helpers\\pydev\\pydevd.py']
    common_file_name = "youtube-comment-analysis-app-python"
    filtered_filenames = [filename for filename in call_filenames if filename.endswith(".py") and common_file_name in filename]
    if len(filtered_filenames)==0:
        return "Not Found"
    caller_filename = os.path.basename(filtered_filenames[-1])
    return caller_filename

source_file_name = get_caller_file_name()
print(source_file_name)


if source_file_name == "OAuth2Security.py":
    filePath = "../params.yaml"
elif source_file_name == "ExtractProperty.py":
    filePath = "../../params.yaml"
elif source_file_name == "main.py":
    filePath = "params.yaml"
else:
    filePath = "params.yaml"


class Property:
    def __init__(self):
        with open(filePath, "r") as yamlfile:
            self.data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    def get_property_data(self):
        return self.data

if __name__ == "__main__":
    # properties = Property()
    # print(properties.get_property_data())
    import os

    print(os.environ['USER'])
    print(os.environ.get('USER'))