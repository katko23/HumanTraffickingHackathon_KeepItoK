from lxml import html

file_path = 'example.txt'  # Replace 'your_file.txt' with the path to your text file
file_content = ""
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        # print(file_content)
except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

def parsing_telegram(file_content):
    # Replace the html_string with your actual HTML content
    html_string = file_content

    # Parse the HTML content
    tree = html.fromstring(html_string)

    # Define the base XPath expression for the parent div
    base_xpath_expression = '//*[@id="MiddleColumn"]/div[4]/div[2]/div/div[1]/div'

    # Extract text content from the parent div and its descendants
    extracted_content = tree.xpath(f'{base_xpath_expression}//text()')

    # Print the extracted content
    print("Extracted Content:")
    for content in extracted_content:
        print(content.strip())

    return extracted_content

parsing_telegram(file_content)