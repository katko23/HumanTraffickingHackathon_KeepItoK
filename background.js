chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === 'get_html') {
    chrome.tabs.executeScript(
      { code: 'document.documentElement.outerHTML' },
      function(result) {
        if (chrome.runtime.lastError) {
          console.error(chrome.runtime.lastError.message);
        } else {
          sendResponse({ html: result[0] });
        }
      }
    );
    return true;  // Indicates that the response will be sent asynchronously
  }
});