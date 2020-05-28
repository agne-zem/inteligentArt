var epochRange = $('#epoch-range-id'),
    epochValue = $('#epoch-span-id'),
    stepsRange = $('#steps-range-id'),
    stepsValue = $('#steps-value-id');
    weightRange = $('#content-weight-id')

epochValue.html(epochRange.attr('value'));
stepsValue.html(stepsRange.attr('value'));

stepsRange.on('input', function(){
    stepsValue.html(this.value);
    $('#generate-btn-id').prop('disabled', true);
    hide('#review-container-id');
    show('#generate-text');
});

epochRange.on('input', function(){
    epochValue.html(this.value);
    $('#generate-btn-id').prop('disabled', true);
    hide('#review-container-id');
    show('#generate-text');
});

weightRange.on('input', function () {
    $('#generate-btn-id').prop('disabled', true);
    hide('#review-container-id');
    show('#generate-text');
});

function updateTextInput(val) {
    document.getElementById('content-span-id').value=val;
}