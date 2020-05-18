$(document).ready(function() {
    $('input:radio[name="test"]').change(function() {
        check();
    });
});

function check() {
    disableEverything();
    show('#check-loader');
    let img = $('#test1').val();

    if ($('#test1').prop("checked") === true) {
        img = $('#test1').val();
    }
    if ($('#test2').prop("checked") === true) {
        img = $('#test2').val();
    }
    if ($('#test3').prop("checked") === true) {
        img = $('#test3').val();
    }

    let message = {
        image: img
    }

    console.log("Checking");
    $.post(Flask.url_for("guess_time"), JSON.stringify(message),
        function (response) {
            hide('#check-loader');
            enableEverything();
            let minutes = Math.floor(response.result.time / 60);
            var seconds = Math.floor(response.result.time - minutes * 60);
            $('#check-text-id').text('Generating this image will ' +
                'take about ' + minutes + ' min ' + seconds + " sec.");
            show('#check-text-id');
        });
}