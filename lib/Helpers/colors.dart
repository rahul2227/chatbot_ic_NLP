import 'package:flutter/material.dart';

const MaterialColor primaryBlack = MaterialColor(
  _blackPrimaryValue,
  <int, Color>{
    50: Color(0xFF000000),
    100: Color(0xFF000000),
    200: Color(0xFF000000),
    300: Color(0xFF000000),
    400: Color(0xFF000000),
    500: Color(_blackPrimaryValue),
    600: Color(0xFF000000),
    700: Color(0xFF000000),
    800: Color(0xFF000000),
    900: Color(0xFF000000),
  },
);
const int _blackPrimaryValue = 0xFF000000;

// Helping Links
// https://medium.com/@nickysong/creating-a-custom-color-swatch-in-flutter-554bcdcb27f3
// Further creation of theme
// https://medium.com/@chathurangacpm/how-to-create-a-custom-theme-in-flutter-and-apply-it-across-the-entire-app-2934ffa17cd4

// Colors used for application
const dullGreen = Color(0xFF02FFB3);
const dullViolet = Color(0xFFEF90FF);
const dullWhtie = Color(0xFFFFFFFF);
const dullBlack = Color(0xFF111111);

// Utility class to create custom colors
MaterialColor createMaterialColor(Color color) {
  List strengths = <double>[.05];
  Map<int, Color> swatch = {};
  final int r = color.red, g = color.green, b = color.blue;

  for (int i = 1; i < 10; i++) {
    strengths.add(0.1 * i);
  }
  for (var strength in strengths) {
    final double ds = 0.5 - strength;
    swatch[(strength * 1000).round()] = Color.fromRGBO(
      r + ((ds < 0 ? r : (255 - r)) * ds).round(),
      g + ((ds < 0 ? g : (255 - g)) * ds).round(),
      b + ((ds < 0 ? b : (255 - b)) * ds).round(),
      1,
    );
  }
  return MaterialColor(color.value, swatch);
}
