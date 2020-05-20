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

$('input:radio[name="type"]').change(function() {
    if($('#type1').prop("checked")) {
        $('#content-weight-id').attr('max', 5000).attr('min', 1)
            .attr('step', 1);
    }

    if($('#type2').prop("checked")) {

        $('#content-weight-id').attr('min', 1).attr('max', 30)
            .attr('step', 1);
    }
    $('#generate-btn-id').prop('disabled', true);
    hide('#review-container-id');
    show('#generate-text');
});

function updateTextInput(val) {
    document.getElementById('content-span-id').value=val;
}