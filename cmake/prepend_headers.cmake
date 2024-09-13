function(prepend_headers_to_source FILE HEADERS)
    foreach(HEADER IN LISTS HEADERS)
        # Read the current content of the source file
        file(READ ${FILE} CONTENTS)
        # Create the header text to prepend
        file(READ ${HEADER} HEADER_CONTENTS)
        # Prepend the header to the contents
        set(NEW_CONTENTS "${HEADER_CONTENTS}\n${CONTENTS}")
        # Write the new contents back to the file
        file(WRITE ${FILE} "${NEW_CONTENTS}")
    endforeach()
endfunction()