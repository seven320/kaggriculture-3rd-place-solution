"""Represent empty order slots in the existing strict Rust JSON parser.

The official _parse_order returns None for both [] and SELL with quantity 0.
The Rust rules ignore zero-quantity trades. Do not remove slots: market order
indices are shared by the two players and deleting [] changes the game.
"""
def compatible_actions(actions):
    return [dict(a,market=[['SELL','WHEAT',0] if o==[] else o
                          for o in a.get('market',[])]) for a in actions]
