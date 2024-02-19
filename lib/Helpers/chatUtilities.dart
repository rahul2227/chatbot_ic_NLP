import 'package:chatbot_ic/Helpers/colors.dart';
import 'package:flutter/material.dart';

class ChatStartBoxButton extends StatelessWidget {
  const ChatStartBoxButton({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      height: MediaQuery.of(context).size.width / 2,
      width: MediaQuery.of(context).size.width / 2,
      margin: const EdgeInsets.all(10.0),
      padding: const EdgeInsets.symmetric(horizontal: 15.0),
      decoration: const BoxDecoration(
        color: dullViolet,
        borderRadius: BorderRadius.all(
          Radius.circular(45.0),
        ),
      ),
      child: const Column(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Icon(
            Icons.margin,
            size: 50,
          ),
          Text(
            "When I'll replace coders!",
            style: TextStyle(fontSize: 22.0),
          ),
        ],
      ),
    );
  }
}
