function(tera_check_layering tier forbidden)
  file(GLOB_RECURSE sources CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${PROJECT_SOURCE_DIR}/include/Tera/${tier}/*.h)
  foreach(source IN LISTS sources)
    file(STRINGS ${source} offending REGEX "^#include \"Tera/(${forbidden})/")
    if(offending)
      message(FATAL_ERROR
              "${source}: ${offending} -- the ${tier} tier is below that one")
    endif()
  endforeach()
endfunction()
