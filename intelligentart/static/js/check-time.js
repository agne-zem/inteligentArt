$(document).ready(function() {
    // calls function to check how much time will it
    // take to generate bigger image of user selected test image
    $('input:radio[name="test"]').change(function() {
        check();
    });
});

//function checking generating time
function check() {
    disableEverything();
    show('#check-loader');
    let img = $('#test1').val();

    // identifying which test image the user has chosen
    if ($('#test1').prop("checked") === true) {
        img = $('#test1').val();
    }
    if ($('#test2').prop("checked") === true) {
        img = $('#test2').val();
    }
    if ($('#test3').prop("checked") === true) {
        img = $('#test3').val();
    }

    // creating a message for HTTP post request
    let message = {
        image: img
    }

    // HTTP post request that checks how much time will the generation take
    // displays response (time, rounded) for user
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
