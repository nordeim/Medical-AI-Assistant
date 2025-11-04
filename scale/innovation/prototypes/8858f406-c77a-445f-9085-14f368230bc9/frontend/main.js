// Innovation Prototype Frontend
document.addEventListener('DOMContentLoaded', function() {
    console.log('Innovation Prototype loaded');
    
    // TODO: Add frontend logic
});

// API Functions
async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(endpoint, options);
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}
