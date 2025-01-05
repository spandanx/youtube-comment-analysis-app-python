import yaml

from src.config.CommonVariables import get_caller_file_name



source_file_name = get_caller_file_name()
print(source_file_name)


if source_file_name == "OAuth2Security.py":
    filePath = "../params.yaml"
elif source_file_name == "ExtractProperty.py":
    filePath = "../../params.yaml"
elif source_file_name == "main.py":
    filePath = "params.yaml"
else:
    filePath = "../../params.yaml"


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