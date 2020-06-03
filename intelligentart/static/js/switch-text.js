$(window).resize(function () {
    switchText();
});
$(document).ready(function () {
    switchText();
});

// switches text in homepage according to the size of the screen
function switchText () {
    if($(window).width() > 992) {
        $('.showcase-text').html('Some works that were made using this website. On the left is the generated image,\n' +
            '            on the right the list of images that were combined to create the generated image. First image in the list\n' +
            '            is used for content and others - for result image stylization.');
    }
    else {
        $('.showcase-text').html('Some works that were made using this website. At the top is the generated image,\n' +
    '            bellow - a list of images that were combined to create the generated image. First image in the list\n' +
    '            is used for content and other(s) - for result image stylization.');
    }
}