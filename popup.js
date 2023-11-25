document.addEventListener('DOMContentLoaded', function () {
  document.getElementById("extractButton").addEventListener('click', function () {
    var procentElement = document.querySelector('.procent');
    console.log(procentElement)
    procentElement.style.visibility = 'visible';

    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      // Send a message to the content script to get HTML
      chrome.tabs.sendMessage(tabs[0].id, { action: 'get_html' }, function (response) {
        if (chrome.runtime.lastError) {
          console.error(chrome.runtime.lastError.message);
          alert('Error: Could not establish connection. Please make sure the page is fully loaded.');
          return;
        }

        // Send HTML content to Python server
        fetch('http://localhost:5000/parse_html', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ html: response.html }),
        })
          .then(response => {
            if (!response.ok) {
              throw new Error('Network response was not ok');
            }
            return response.json();
          })
          .then(data => {
            // Display the JSON response on the popup
            alert(JSON.stringify(data));
          })
          .catch(error => {
            console.error('Error:', error);
            alert('Error: Could not connect to the Python server.');
          });
      });
    });
  });
});
