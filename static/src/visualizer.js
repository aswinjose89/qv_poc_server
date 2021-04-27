$( function() {
  $.widget( "custom.iconselectmenu", $.ui.selectmenu, {
    _renderItem: function( ul, item ) {
      var li = $( "<li>" ),
        wrapper = $( "<div>", { text: item.label } );

      if ( item.disabled ) {
        li.addClass( "ui-state-disabled" );
      }

      $( "<span>", {
        style: item.element.attr( "data-style" ),
        "class": "ui-icon " + item.element.attr( "data-class" )
      })
        .appendTo( wrapper );

      return li.append( wrapper ).appendTo( ul );
    }
  });

  $( "#sample_img" )
    .iconselectmenu()
    .iconselectmenu( "menuWidget")
      .addClass( "ui-menu-icons avatar" );
} );

var demo2 = $('.demo2').bootstrapDualListbox({
          preserveSelectionOnMove: 'moved',
          moveOnSelect: false,
          infoTextFiltered:'<span class="badge badge-warning">Filtered</span> {0} from {1}',
        });

$( document ).ready(function() {
    $('.visualizerForm').on('submit',function(e){
        e.preventDefault();
        let uploaded_files= JSON.parse(localStorage.getItem("uploaded_files"));

        var data_file = $("#data_file").val();
        if(data_file && uploaded_files.map(x=>x.file).includes(data_file.split('\\').pop())){
            alert("Same File Has been Uploaded Already. Please select the file from data file list..");
        }
        else{
            var formData=$(this).serialize();
            var fullUrl = window.location.origin+window.location.pathname;
            var finalUrl = fullUrl+"?"+formData;
            window.location.href = finalUrl;
        }
    })
});