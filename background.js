// Handle the click event on the extension icon
chrome.browserAction.onClicked.addListener(function(tab) {
  // Send a message to the content script to get the full HTML
  chrome.tabs.sendMessage(tab.id, { action: 'get_html' }, function(response) {
    // Process the HTML content here
    console.log('HTML content:', response.html);
  });
});