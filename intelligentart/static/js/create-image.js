function hide(id) {
    if(!$(id).hasClass('hide')){
    $(id).addClass('hide');
 }
}

function show(id) {
    if ($(id).hasClass('hide')) {
        $(id).removeClass('hide');
    }
}

function disableEverything() {
    $('.input-button').prop('disabled', true);
    $('#type1').prop('disabled', true);
    $('#type2').prop('disabled', true);
    $('.input-range').prop('disabled', true);
    $('#test-btn-id').prop('disabled', true);
    $('#generate-btn-id').prop('disabled', true);
    $('.remove-button').prop('disabled', true);
    $('#review-btn-id').prop('disabled', true);
}

function enableEverything() {
    $('.input-button').prop('disabled', false);
    $('#type1').prop('disabled', false);
    $('#type2').prop('disabled', false);
    $('.input-range').prop('disabled', false);
    $('#test-btn-id').prop('disabled', false);
    $('#generate-btn-id').prop('disabled', false);
    $('.remove-button').prop('disabled', false);
    $('#review-btn-id').prop('disabled', false);
}

//removes image with button click
function removeImage(image_id, button_id, input_id) {
    $(image_id).attr('src', '');
    $(input_id).val('');
    hide(image_id)
    hide(button_id);
    $('#generate-btn-id').prop('disabled', true);
    hide('#review-container-id');
}

//file reader function
function readFile(input, img_id) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        const fileType = input.files[0]['type'];
        const validImageTypes = ['image/jpg', 'image/jpeg', 'image/png'];
        if (!validImageTypes.includes(fileType)) {
            alert("Only jpg png jpeg.");
            $(img_id).attr('src', "");
            return
        }
        reader.onload = function (e) {
            let dataURL = reader.result;
            $(img_id).attr('src', dataURL);
        };

        reader.readAsDataURL(input.files[0]);
    }
    show(img_id);
}

//gets selected style layer names in array
function getStyleLayers() {
    let style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1']

    if ($('#type1').prop("checked") === true) {
        style_layers.push('block4_conv1')
        style_layers.push('block5_conv1')
    }

    return style_layers;
}

//gets selected content layer
function getContentLayer() {
    let content_layer = [];

    if ($('#type2').prop("checked") === true) {
        content_layer.push('block5_conv2')
    }

    else {
        content_layer.push('block5_conv2')
    }

    return content_layer;
}

//removes padding from dataURL
function removePadding(img_id) {
    let data = document.getElementById(img_id).getAttribute('src')
    let result = data.replace(/^data:image\/(png|jpg|jpeg);base64,/, "");
    return result;
}

//checks if image src is empty or not
function isImageEmpty(img_id) {
    let result = true;
    if (document.getElementById(img_id).getAttribute('src') !== '') {
        result = false;
    }
    return result;
}

//gets style images into array
function getStyleImagesArray() {
    let style_images = [];
    for (var i = 1; i < 5; i++) {
        let img_id = 'style-img-id' + i.toString();
        if(!isImageEmpty(img_id)) {
            let data = removePadding(img_id);
            style_images.push(data);
        }
    }
    return style_images;
}

function validation() {
    passed = true;
    //checking if content image is selected
    if (isImageEmpty('content-img-id')) {
        alert('No content image selected');
        passed = false;
    }

    //checking if at least one style image is selected
    if (isImageEmpty('style-img-id1') &&
        isImageEmpty('style-img-id2') &&
        isImageEmpty('style-img-id3') &&
        isImageEmpty('style-img-id4')) {
        alert('At least one style image must be selected');
        passed = false;
    }

    return passed;
}

function selectRating() {
    let val = $('[name="rating"]:checked').val();
    return val;
}

function getChosenTest() {
    let img;
    if ($('#test1').prop("checked") === true) {
        img = $('#test1').val();
    }
    if ($('#test2').prop("checked") === true) {
        img = $('#test2').val();
    }
    if ($('#test3').prop("checked") === true) {
        img = $('#test3').val();
    }
    return img;
}

//main
$(document).ready(function() {
    let styleDisplay = 1;
    $("#content-img-upload").change(function () {
        readFile(this, '#content-img-id');
        $('#generate-btn-id').prop('disabled', true);
        show('#generate-text');
    });

    $("#style-img-upload1").change(function () {
        readFile(this, '#style-img-id1');
        $('#generate-btn-id').prop('disabled', true);
        show('#generate-text');
        show('#remove1');
    });

    $("#style-img-upload2").change(function () {
        readFile(this, '#style-img-id2');
        $('#generate-btn-id').prop('disabled', true);
        show('#generate-text');
        show('#remove2');
    });

    $("#style-img-upload3").change(function () {
        readFile(this, '#style-img-id3');
        $('#generate-btn-id').prop('disabled', true);
        show('#generate-text');
        show('#remove3');
    });

    $("#style-img-upload4").change(function () {
        readFile(this, '#style-img-id4');
        $('#generate-btn-id').prop('disabled', true);
        show('#generate-text');
        show('#remove4');
    });

    $('#test-btn-id').click(function (event) {
        if(!validation()) {
            return;
        }
        disableEverything();
        hide('#generate-container');
        hide('#generate-container-id');
        hide('#review-popup-container');
        hide('#result-container-id');
        hide('#check-text-id');
        hide('.select-test-text');
        hide('#generate-btn-id');
        hide('#generate-text');
        hide('#testing-images-container-id');
        show('#test-loader');
        show('#test-loader-text');

        $("body,html").animate({
            scrollTop: $("#testing-container-id").offset().top
        },800);

        let content_image = removePadding('content-img-id');
        let style_images = getStyleImagesArray();
        let epochs = $('#epoch-range-id').val();
        let steps =  $('#steps-range-id').val();
        let content_weight =  $('#content-weight-id').val();
        console.log($('#content-weight-id').val());
        let style_layers = getStyleLayers();
        let content_layer = getContentLayer();

        let message = {
            content_image: content_image,
            style_images: style_images,
            epochs: epochs,
            steps: steps,
            content_weight : content_weight,
            style_layers: style_layers,
            content_layer: content_layer

        }
        console.log("Testing");
        $.post(Flask.url_for("test"), JSON.stringify(message),
            function (response) {
                enableEverything();
                hide('#test-loader');
                hide('#test-loader-text');
                hide('#thank-you-id');
                show('#testing-images-container-id');
                show('.select-test-text');
                $('#generate-btn-id').prop('disabled', false);
                hide('#generate-text');
                show('#generate-container-id');
                $('#test-generated1-img-id').attr('src', "/static/generated/" + response.result.test_images[0])
                    .show();

                $('#test-generated2-img-id').attr('src', "/static/generated/" + response.result.test_images[1])
                    .show();
                $('#test-generated3-img-id').attr('src', "/static/generated/" + response.result.test_images[2])
                    .show();

                $('#test1').attr('value', response.result.test_images[0]);
                $('#test2').attr('value', response.result.test_images[1]);
                $('#test3').attr('value', response.result.test_images[2]);
                check()
                show('#generate-btn-id');
                $("body,html").animate({
                    scrollTop: $("#testing-container-id").offset().top
                },800);
            });
    });

    $('#generate-btn-id').click(function (event) {
        if(!validation()) {
            return;
        }

        show('#result-container-id');
        hide('#review-container-id');
        show('#loader');
        show('#result-container-id');
        disableEverything();
        $("body,html").animate({
            scrollTop: $("#result-container-id").offset().top
          },800);

        let image = getChosenTest();

        let message = {
            file_name: image
        }

        console.log("Generating");
        $.post(Flask.url_for("generate"), JSON.stringify(message),
            function (response) {
                enableEverything();
                show('#review-container-id');
                hide('#loader');
                show('#save-container-id');
                show('#review-btn-id');
                $('#save-id').attr('href', "/static/generated/" + response.result.generated_image)
                    .attr('download', Date.now());
                $('#generated-img-id').attr('src', "/static/generated/" + response.result.generated_image)
                    .attr('alt', response.result.generated_image);
                $("body,html").animate({
                    scrollTop: $("#result-container-id").offset().top
                },800);
            });
    });


    $('#review-btn-id').click(function (event) {
        let file_name = $('#generated-img-id').attr('alt');
        let rating = selectRating();
        let message = {
            file_name: file_name,
            rating: rating
        }
        disableEverything();
        show('#review-loader');
        console.log('rating');
        $.post(Flask.url_for("review"), JSON.stringify(message),
            function (response) {
                show('#thank-you-id');
                hide('#review-loader');
                hide('#review-btn-id');
                enableEverything();
            });
    });
});