
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === 'get_html') {
    sendResponse({ html: document.documentElement.outerHTML });
  }
  return true; // Remove this line
});