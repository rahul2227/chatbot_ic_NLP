import 'package:chatbot_ic/Helpers/chatUtilities.dart';
import 'package:chatbot_ic/Helpers/colors.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // colorScheme: ColorScheme.fromSeed(seedColor: primaryBlack),
        // primaryColor: Colors.black,
        primarySwatch: createMaterialColor(dullBlack),
        scaffoldBackgroundColor: createMaterialColor(dullBlack),
        useMaterial3: false,
      ),
      home: const MyHomePage(title: 'Chatbot'),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  List boxList = [
    ChatStartBoxButton(),
    ChatStartBoxButton(),
    ChatStartBoxButton(),
    ChatStartBoxButton(),
  ];
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        // backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(
          widget.title,
          style: const TextStyle(color: dullWhtie),
        ),
      ),
      body: ListView(
        // mainAxisAlignment: MainAxisAlignment.spaceAround,
        // crossAxisAlignment: CrossAxisAlignment.start,
        padding: const EdgeInsets.all(8.0),
        scrollDirection: Axis.vertical,
        children: <Widget>[
          SizedBox(
            height: MediaQuery.of(context).size.width / 2,
            child: ListView(
                scrollDirection: Axis.horizontal,
                children: <Widget>[...boxList]),
          ),
          SizedBox(
            height: MediaQuery.of(context).size.width / 2,
            child: ListView(
                scrollDirection: Axis.horizontal,
                children: <Widget>[...boxList]),
          ),
          SizedBox(
            height: MediaQuery.of(context).size.width / 2,
            child: ListView(
                scrollDirection: Axis.horizontal,
                children: <Widget>[...boxList]),
          ),
          SizedBox(
            height: MediaQuery.of(context).size.width / 2,
            child: ListView(
                scrollDirection: Axis.horizontal,
                children: <Widget>[...boxList]),
          ),
        ],
      ),
    );
  }
}
