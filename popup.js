document.getElementById('extractButton').addEventListener('click', function () {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    chrome.tabs.sendMessage(tabs[0].id, { action: 'get_html' }, function (response) {
      if (chrome.runtime.lastError) {
        console.error(chrome.runtime.lastError.message);
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
        .then(response => response.json())
        .then(data => {
          // Display the JSON response on the popup
          alert(JSON.stringify(data));
        })
        .catch(error => {
          console.error('Error:', error);
        });
    });
  });
});