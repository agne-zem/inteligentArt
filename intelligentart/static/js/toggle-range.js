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
    show('#generate-text');
});

epochRange.on('input', function(){
    epochValue.html(this.value);
    $('#generate-btn-id').prop('disabled', true);
    show('#generate-text');
});

weightRange.on('input', function () {
    $('#generate-btn-id').prop('disabled', true);
    show('#generate-text');
});

$('input:radio[name="type"]').change(function() {
    if($('#type1').prop("checked")) {
        $('#content-weight-id').attr('max', 1000000).attr('min', 10)
            .attr('step', 20000).attr('value', 10);
    }

    if($('#type2').prop("checked")) {

        $('#content-weight-id').attr('max', 30)
            .attr('step', 2).attr('value', 1);
    }
    $('#generate-btn-id').prop('disabled', true);
    show('#generate-text');
});