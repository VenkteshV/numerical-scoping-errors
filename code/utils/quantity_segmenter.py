"""Quantity segmenter that returns the quantitiative segments within a claim

    'cd D:\Programs\stanford-corenlp\stanford-corenlp-4.5.10'
    First run CoreNLP server with: 'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000'

    Returns:
        List[str]: List of quantitative segments
"""
import json
import re
from pycorenlp import StanfordCoreNLP
from typing import Any, List, Tuple

from collections import deque


class Node():
    """encodes node of the constituency parse tree
    """

    def __init__(self, value=None) -> None:
        self.value = value
        self.children = []

    def __eq__(self, other: Any) -> bool:
        """ Check equality of node values
        Overrides the default implementation
        For our purposes, at the node level, we consider a node equal
        to another if they share the same value
        and their children share the same value"""
        if isinstance(other, Node):
            is_equal = self.value == other.value and len(
                self.children) == len(other.children)
            if is_equal:
                for child_ind in range(len(self.children)):
                    if self.children[child_ind].value != other.children[child_ind].value:
                        return False
                return is_equal
        return False

    def __ne__(self, other: Any) -> bool:
        """Check equality of node values

        Args:
            other (Any): Node value

        Returns:
            bool: indication of whether nodes are equal
        """
        return not self.__eq__(other)

    def set_value(self, value):
        self.value = value

    def set_child_at_index(self, child, index=-1):
        if index == -1:
            self.children.append(child)
        else:
            self.children[index] = child

    def get_value(self):
        return self.value

    def get_child_at_index(self, index=-1):
        assert abs(index) < len(self.children)
        return self.children[index]

    def get_children(self):
        return self.children


class Tree():
    def __init__(self, root):
        '''
        Initialize tree with root
        @param root: root is an object of type Node, representing root 
        of the binary tree
        '''
        self.root = root

    def __eq__(self, other):
        """Overrides the default implementation
        For our purposes, at the node level, we consider a node equal 
        to another if they share the same value
        and their children share the same value"""
        if isinstance(other, Tree):
            return self.get_prefix_traversal(
                self.root) == self.get_prefix_traversal(other.root)
        return False

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        return not self.__eq__(other)

    def get_prefix_traversal(self, start_node=None):
        '''
        Get prefix traversal of tree
        :param start_node: node in the tree, which is root of the subtree we
        would like a prefix traversal for
        :return: prefix traversal
        '''

        if start_node == None:
            return []

        def is_pos_tag(node):
            children = node.get_children()
            return len(children) == 1 and children[0] and len(children[0].get_children()) == 0

        # If we reach a POS tag we follow the same format as
        # the Stanford parser prefix
        if is_pos_tag(start_node):
            prefix_traversal = [
                "(", start_node.value, start_node.get_children()[0].value, ")"]
            return prefix_traversal

        prefix_traversal = ["(", start_node.value]

        for each_child in start_node.get_children():
            prefix_traversal += self.get_prefix_traversal(each_child)

        prefix_traversal.append(")")
        return prefix_traversal

    def get_ancestors(self, query_node, current_node, path=[]):
        '''
        Get all ancestors along the path from root to a query node
        :param query_node: Node object whose ancestors we want
        '''

        if current_node == None:
            return []

        # If we reach our query node
        if current_node == query_node:
            return [current_node]

        found_query = False
        for each_child in current_node.get_children():
            path_to_leaf = self.get_ancestors(query_node, each_child, path)

            if len(path_to_leaf) > 0:
                found_query = True
                path = path_to_leaf
                path.append(current_node)
                return path

        if not found_query:
            return []

    def get_nodes_with_value(self, query_value, start_node, path=[]):
        '''
        Get all nodes with a particular value
        :param query_node: Node object whose ancestors we want
        '''
        nodes_with_value = []

        def visit_all_nodes(query_value, current_node):

            if current_node == None:
                return

            # If we reach our query node
            if current_node.value == query_value:
                nodes_with_value.append(current_node)

            for each_child in current_node.get_children():
                visit_all_nodes(query_value, each_child)

        visit_all_nodes(query_value, start_node)
        return nodes_with_value


def get_subtree(parts, start_index):
    '''
    get_subtree returns the end_index in the parse prefix expression
    corresponding to a given subtree
    :param parts: prefix expression of constituency parse of the sentence
    in the form of a list
    :param start_index: start of the subtree
    :return: end_index: end of the subtree in the prefix expression
    '''

    # This is invalid
    if parts[start_index] != '(':
        return -1

    # Create a deque to use it as a stack.
    stack = deque()

    for element_index in range(start_index, len(parts)):

        # Pop a starting bracket
        # for every closing bracket
        if parts[element_index] == ')':
            stack.popleft()

        # Push all starting brackets
        elif parts[element_index] == '(':
            stack.append(parts[element_index])

        # If stack becomes empty
        if not stack:
            end_index = element_index
            return end_index

    return -1


def get_tree(parts: List[Any], start_index: int) -> Node:
    """returns tree structure

    Args:
        parts (Node): subtree
        start_index (_type_): start index of root node

    Returns:
        Node: returns tree structure
    """

    if start_index >= len(parts):
        return Node(None)

    # Current Node in prefix expression becomes value of the node
    value_index = start_index + 1  # TAG after opening brace
    value = parts[value_index]
    node = Node(value)

    subtree_start = value_index + 1
    while parts[subtree_start] == "(":
        subtree_end = get_subtree(parts, subtree_start)
        if subtree_end - subtree_start == 3:
            pos_tag = parts[subtree_start + 1]
            word = parts[subtree_end - 1]

            child_node = Node(pos_tag)
            word_node = Node(word)

            child_node.set_child_at_index(word_node, -1)
            node.set_child_at_index(child_node, -1)
        else:
            node.set_child_at_index(get_tree(parts, subtree_start), -1)
        subtree_start = subtree_end + 1

    return node


def construct_parse(parse: str) -> Tree:
    """Converts prefix expression of parse to an expression tree

    Args:
        parse (str): constituency parse of the sentence in prefix

    Returns:
        Tree: expression tree of constituency parse of the sentence
    """
    parse = parse.replace("(", " ( ")
    parse = parse.replace(")", " ) ")
    parts = parse.split()
    start_index = 0

    root = get_tree(parts, start_index)
    parse_tree = Tree(root)

    return parse_tree


class Segmenter():
    """
    Inputs:
        args:
            sentence (str) : sentence to extract quantity mentions from
    Outputs:
        quantity mentions (list[str]) : list of noun phrases
        containing quantities from parse
    """

    def __init__(self):
        # pass
        self.nlp = StanfordCoreNLP('http://localhost:9000')

    def normalize_text(self, text) -> str:
        """
        Normalizes input text to segmenter. Inserts ' ' between numbers and currency codes and normalizes quotes.

        Args:
            text (str): Text we want to normalize

        Returns:
            Normalized text
        """
        # Think tags
        text = re.sub(r"</?think>", " ", text)
        # Weird quotes
        text = (text
            .replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"')
        )
        # Space between letters and digits both directions:
        text = re.sub(r"(?<=[A-Za-z$])(?=\d)", " ", text)
        text = re.sub(r"(?<=\d)(?=[A-Za-z%])", " ", text)
        # Separate %
        text = re.sub(r"(?<=\d)%(?=\b)", " %", text)
        # Extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


    def segment(self, text) -> Tuple[List[List[str]], List[Node]]:
        '''Extracts and returns all quantity mentions in a text.
        Quantity mentions are defined as least ancestor noun phrases
        containing quantities.
        Args:
        text (str): Sentence (typically) that we want to extract 
        quantity mentions from
        Returns:
          quantity_mentions, noun_phrases: List of quantity mentions 
          found in sentence'''

        noun_phrases_all, quantity_phrases, quantity_natural_language = [], [], []
        parse = self.nlp.annotate(text, properties={
            'annotators': 'parse',
            'outputFormat': 'json',
            'timeout': 10000,
        })
        
        # pycorenlp may return a raw JSON string; convert it.
        if isinstance(parse, str):
            # Helpful debug: see what the server actually returned (first 300 chars)
            # print("CoreNLP raw response (first 300 chars):", parse[:300])

            try:
                parse = json.loads(parse)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    "CoreNLP did not return valid JSON. "
                    "The server likely returned an error message instead.\n\n"
                    f"Raw response (first 1000 chars):\n{parse[:1000]}"
                ) from e

        for i, sentence in enumerate(parse['sentences']):
            each_parse_sentence = sentence['parse']
            parse_tree = construct_parse(each_parse_sentence)
            sent_text = " ".join(tok['word'] for tok in sentence.get("tokens", []))
            quantities = parse_tree.get_nodes_with_value("CD", parse_tree.root)
            
            ancestor_chains = [parse_tree.get_ancestors(
                quantity, parse_tree.root) for quantity in quantities]
            
            seen_spans = set()

            for each_chain, quantity in zip(ancestor_chains, quantities):
                chain_values = [node.value for node in each_chain]
                qp_index = chain_values.index("QP") if "QP" in chain_values else -1
                np_index = chain_values.index("NP") if "NP" in chain_values else -1

                # FInds least common NP-TMP phrase
                if "NP-TMP" in chain_values:
                    tmp_index = chain_values.index("NP-TMP")
                    if np_index == -1 or tmp_index < np_index:
                        np_index = tmp_index

                candidate = None

                if qp_index != -1 and np_index != -1:
                    qp_node = each_chain[qp_index]
                    np_node = each_chain[np_index]

                    # scan NP subtree for noun POS
                    stack = [np_node]
                    np_has_noun_pos = False
                    np_has_percent_token = False
                    while stack:
                        n = stack.pop()
                        if n.value in {'NN', "NNS", "NNP", "NNPS"}:
                            np_has_noun_pos = True
                        if len(n.children) == 0 and isinstance(n.value, str) and n.value == "%":
                            np_has_percent_token = True
                        stack.extend(n.children)

                    # scan QP subtree for noun POS (to see if NP adds extra noun context)
                    stack = [qp_node]
                    qp_has_noun_pos = False
                    while stack:
                        n = stack.pop()
                        if n.value in {"NN", "NNS", "NNP", "NNPS"}:
                            qp_has_noun_pos = True
                        stack.extend(n.children)

                    # Decision:
                    # - If NP contains '%' token, prefer NP (keeps "50 %")
                    # - Else if NP has a noun POS and QP does not, prefer NP (keeps "seven times", "29 seasons")
                    # - Else prefer QP (keeps tight "18 billion", "38 billion", etc.)
                    if np_has_percent_token:
                        candidate = np_node
                    elif np_has_noun_pos and not qp_has_noun_pos:
                        candidate = np_node
                    else:
                        candidate = qp_node
                elif np_index != -1:
                    candidate = each_chain[np_index]
                elif qp_index != -1:
                    candidate = each_chain[qp_index]

                if candidate is not None:
                    span_list = parse_tree.get_prefix_traversal(candidate)
                    span = " ".join(span_list)
                    if span not in seen_spans:
                        seen_spans.add(span)
                        quantity_phrases.append(span_list)
                        noun_phrases_all.append(candidate)

                        # Extract leaf tokens under candidate to make natural language output
                        leaf_tokens = []
                        stack = [candidate]
                        while stack:
                            n = stack.pop()
                            if len(n.children) == 0 and isinstance(n.value, str):
                                leaf_tokens.append(n.value)
                            else:
                                # preserve L-R order
                                stack.extend(reversed(n.children))
                        quantity_text = " ".join(leaf_tokens)

                        # Extract leaf tokens under quantity to make the numerical output
                        leaf_tokens = []
                        stack = [quantity]
                        while stack:
                            n = stack.pop()
                            if len(n.children) == 0 and isinstance(n.value, str):
                                leaf_tokens.append(n.value)
                            else:
                                # preserve L-R order
                                stack.extend(reversed(n.children))
                        quantity = " ".join(leaf_tokens)
                        quantity_natural_language.append({'quantity_text': quantity_text, 'quantity': quantity, 'sentence_idx': i, "sentence_text": sent_text})

        return quantity_phrases, noun_phrases_all, quantity_natural_language


if __name__ == "__main__":
    """Example quantity segment identification
    """
    segmenter = Segmenter()
    quantity_phrases, noun_phrases, quantity_natural_language = segmenter.segment("""During his 29
      seasons as chairman of the chicago bulls ,
      the team captured the title seven times ,
      including in 2000""")
    print("quantity_phrases", quantity_phrases)