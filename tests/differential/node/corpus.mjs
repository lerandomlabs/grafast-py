// Phase C — SHARED differential corpus (reference / Node side).
//
// This file is the canonical-TS-Grafast encoding of the corpus. Its Python twin is
// gtests/diff_corpus.py — the two MUST encode the same fixtures (same `name`, same
// SDL text, same seed values, matched plan wiring, byte-identical query + vars).
// The differ asserts the fixture-name SET is identical across the two result files,
// so any drift in coverage is itself a hard failure.
//
// A fixture is { name, sdl, plans, query, variables }. `plans` is the
// makeGrafastSchema `objects` map. Batch-load callbacks are referenced by stable
// names so the harness can wrap them with a fetch counter; loaders a fixture never
// triggers simply do not appear in its fetchCounts (both sides agree on absence).

import {
  constant,
  access,
  lambda,
  list,
  object,
  each,
  loadOne,
  loadMany,
} from "grafast";

// --------------------------------------------------------------------- seed data
// Plain JSON-able values, written identically in diff_corpus.py. Module scope so
// plan closures can read them.

const AUTHORS = [
  { id: 1, name: "Ada", bio: null, tags: ["math", "engine"] },
  { id: 2, name: "Babbage", bio: "engines", tags: [] },
  { id: 3, name: "Curie", bio: "radioactivity", tags: ["physics"] },
];

const AUTHOR_BY_ID = { 1: AUTHORS[0], 2: AUTHORS[1], 3: AUTHORS[2] };

// posts per author id; author 3 has no posts (empty-list fixture). Post.title is
// NON-NULL in the SDL; post 99 carries a null title to drive null-bubbling.
const POSTS_BY_AUTHOR = {
  1: [
    { id: 11, title: "A1", authorId: 1 },
    { id: 12, title: "A2", authorId: 1 },
  ],
  2: [{ id: 21, title: "B1", authorId: 2 }],
  3: [],
};

const ALL_POSTS = [
  { id: 11, title: "A1", authorId: 1 },
  { id: 12, title: "A2", authorId: 1 },
  { id: 21, title: "B1", authorId: 2 },
];

// comments per post id (for deep nesting).
const COMMENTS_BY_POST = {
  11: [{ id: 111, body: "c-a" }, { id: 112, body: "c-b" }],
  12: [{ id: 121, body: "c-c" }],
  21: [{ id: 211, body: "c-d" }],
};

// coauthor id lists (for each + loadOne).
const COAUTHOR_IDS = { 1: [2, 3], 2: [1], 3: [] };

// --------------------------------------------------------- named batch callbacks
// Pure functions; the harness wraps these with a counter keyed by the given name.
// Each takes the array of ALL lookup keys in the bucket and returns an index-aligned
// array (loadOne: one record per key; loadMany: one sub-array per key).

export const LOADERS = {
  loadAuthors: (ids) => ids.map((id) => AUTHOR_BY_ID[id] ?? null),
  loadPostsByAuthor: (ids) => ids.map((id) => POSTS_BY_AUTHOR[id] ?? []),
  loadCommentsByPost: (ids) => ids.map((id) => COMMENTS_BY_POST[id] ?? []),
};

// ------------------------------------------------------------------------- SDLs
const SDL_BLOG = /* GraphQL */ `
  type Query {
    hello: String
    answer: Int
    flag: Boolean
    me: Author
    authors: [Author!]!
    author(id: Int!): Author
    posts: [Post!]!
    echo(n: Int): Int
    color(pick: Hue!): String
    greet(who: Name!): String
    sum(xs: [Int!]!): Int
    boom: String
  }
  type Author {
    id: Int!
    name: String!
    bio: String
    tags: [String!]!
    posts: [Post!]!
    coauthors: [Author!]!
    title: String!
    nullTags: [String!]!
  }
  type Post {
    id: Int!
    title: String!
    author: Author
    comments: [Comment!]!
  }
  type Comment {
    id: Int!
    body: String!
  }
  enum Hue { RED GREEN BLUE }
  input Name { first: String! last: String! }
`;

// ------------------------------------------------------------ shared plan pieces
// Leaf plans read a key off the parent step. access($s, "id") on the JS side.
const leaf = (key) => ($p) => access($p, key);

// Build the `objects` plan map for a fixture from a partial set of Query plans,
// re-using the shared Author/Post/Comment plans so every fixture has consistent
// nested wiring.
function objectsFor(queryPlans) {
  return {
    Query: { plans: queryPlans },
    Author: {
      plans: {
        id: leaf("id"),
        name: leaf("name"),
        bio: leaf("bio"),
        tags: leaf("tags"),
        // NON-NULL String field whose value is always null in the data -> bubbles.
        title: ($a) => constant(null),
        // [String!]! whose first element is null -> element bubbles, list NonNull.
        nullTags: ($a) => constant(["ok", null]),
        posts: ($a) => loadMany(access($a, "id"), LOADERS.loadPostsByAuthor),
        coauthors: ($a) => {
          const $ids = lambda(access($a, "id"), (aid) => COAUTHOR_IDS[aid]);
          return each($ids, ($id) => loadOne($id, LOADERS.loadAuthors));
        },
      },
    },
    Post: {
      plans: {
        id: leaf("id"),
        title: leaf("title"),
        author: ($p) => loadOne(access($p, "authorId"), LOADERS.loadAuthors),
        comments: ($p) => loadMany(access($p, "id"), LOADERS.loadCommentsByPost),
      },
    },
    Comment: {
      plans: { id: leaf("id"), body: leaf("body") },
    },
  };
}

// ----------------------------------------------------------------- the fixtures
export const FIXTURES = [
  {
    name: "flat_scalars",
    sdl: SDL_BLOG,
    plans: objectsFor({
      hello: () => constant("world"),
      answer: () => constant(42),
      flag: () => constant(true),
    }),
    query: `{ hello answer flag }`,
    variables: {},
  },
  {
    name: "nested_object",
    sdl: SDL_BLOG,
    plans: objectsFor({ me: () => constant(AUTHORS[0]) }),
    query: `{ me { id name } }`,
    variables: {},
  },
  {
    name: "list_of_objects",
    sdl: SDL_BLOG,
    plans: objectsFor({ authors: () => constant(AUTHORS) }),
    query: `{ authors { id name } }`,
    variables: {},
  },
  {
    name: "list_of_leaves",
    sdl: SDL_BLOG,
    plans: objectsFor({
      author: (_p, args) => loadOne(args.getRaw("id"), LOADERS.loadAuthors),
    }),
    query: `{ author(id: 1) { tags } }`,
    variables: {},
  },
  {
    name: "deep_nesting",
    sdl: SDL_BLOG,
    plans: objectsFor({ authors: () => constant(AUTHORS) }),
    query: `{ authors { posts { comments { body } } } }`,
    variables: {},
  },
  {
    name: "null_leaf_nullable",
    sdl: SDL_BLOG,
    plans: objectsFor({
      author: (_p, args) => loadOne(args.getRaw("id"), LOADERS.loadAuthors),
    }),
    query: `{ author(id: 1) { bio } }`,
    variables: {},
  },
  {
    name: "null_in_nonnull_list",
    sdl: SDL_BLOG,
    plans: objectsFor({
      author: (_p, args) => loadOne(args.getRaw("id"), LOADERS.loadAuthors),
    }),
    query: `{ author(id: 1) { nullTags } }`,
    variables: {},
  },
  {
    name: "nonnull_field_returns_null",
    sdl: SDL_BLOG,
    plans: objectsFor({
      author: (_p, args) => loadOne(args.getRaw("id"), LOADERS.loadAuthors),
    }),
    query: `{ author(id: 1) { title } }`,
    variables: {},
  },
  {
    name: "nonnull_bubbles_to_root",
    sdl: SDL_BLOG,
    plans: objectsFor({
      author: (_p, args) => loadOne(args.getRaw("id"), LOADERS.loadAuthors),
    }),
    // author(id:1).title is String! resolving null; with author nullable the bubble
    // stops at author, so to bubble to root we select a NonNull root field. We use a
    // NonNull wrapper via the schema: `me` is nullable, so instead drive root-bubble
    // through `author` whose `title` is NonNull but `author` itself is nullable.
    // True root bubble: query a NonNull list element going null is overkill; use the
    // dedicated path below. Here we assert the data:null root case via authors[0].title
    // inside a NonNull list:
    query: `{ authors { title } }`,
    variables: {},
  },
  {
    name: "aliases",
    sdl: SDL_BLOG,
    plans: objectsFor({
      hello: () => constant("world"),
      answer: () => constant(42),
      me: () => constant(AUTHORS[0]),
    }),
    query: `{ a: hello b: hello x: answer who: me { ident: id label: name } }`,
    variables: {},
  },
  {
    name: "arg_scalar",
    sdl: SDL_BLOG,
    plans: objectsFor({
      echo: (_p, args) => lambda(args.getRaw("n"), (n) => n),
    }),
    query: `{ echo(n: 7) }`,
    variables: {},
  },
  {
    name: "arg_enum",
    sdl: SDL_BLOG,
    plans: objectsFor({
      color: (_p, args) => lambda(args.getRaw("pick"), (pick) => `picked:${pick}`),
    }),
    query: `{ color(pick: GREEN) }`,
    variables: {},
  },
  {
    name: "arg_input_object",
    sdl: SDL_BLOG,
    plans: objectsFor({
      greet: (_p, args) =>
        lambda(args.getRaw("who"), (who) => `${who.first} ${who.last}`),
    }),
    query: `{ greet(who: { first: "Ada", last: "Lovelace" }) }`,
    variables: {},
  },
  {
    name: "arg_list",
    sdl: SDL_BLOG,
    plans: objectsFor({
      sum: (_p, args) =>
        lambda(args.getRaw("xs"), (xs) => xs.reduce((a, b) => a + b, 0)),
    }),
    query: `{ sum(xs: [1, 2, 3]) }`,
    variables: {},
  },
  {
    name: "var_required",
    sdl: SDL_BLOG,
    plans: objectsFor({
      author: (_p, args) => loadOne(args.getRaw("id"), LOADERS.loadAuthors),
    }),
    query: `query Q($id: Int!) { author(id: $id) { name } }`,
    variables: { id: 2 },
  },
  {
    name: "var_default",
    sdl: SDL_BLOG,
    plans: objectsFor({
      echo: (_p, args) => lambda(args.getRaw("n"), (n) => n),
    }),
    query: `query Q($n: Int = 5) { echo(n: $n) }`,
    variables: {},
  },
  {
    name: "fragment_spread",
    sdl: SDL_BLOG,
    plans: objectsFor({ me: () => constant(AUTHORS[1]) }),
    query: `{ me { ...F } } fragment F on Author { id name bio }`,
    variables: {},
  },
  {
    name: "inline_fragment",
    sdl: SDL_BLOG,
    plans: objectsFor({ me: () => constant(AUTHORS[1]) }),
    query: `{ me { id ... on Author { bio } } }`,
    variables: {},
  },
  {
    name: "skip_true",
    sdl: SDL_BLOG,
    plans: objectsFor({ me: () => constant(AUTHORS[0]) }),
    query: `query Q($s: Boolean!) { me { id name @skip(if: $s) } }`,
    variables: { s: true },
  },
  {
    name: "include_false",
    sdl: SDL_BLOG,
    plans: objectsFor({ me: () => constant(AUTHORS[0]) }),
    query: `query Q($i: Boolean!) { me { id name @include(if: $i) } }`,
    variables: { i: false },
  },
  {
    name: "loadmany_n_plus_1",
    sdl: SDL_BLOG,
    plans: objectsFor({ authors: () => constant(AUTHORS) }),
    query: `{ authors { posts { id } } }`,
    variables: {},
  },
  {
    name: "loadone_n_plus_1",
    sdl: SDL_BLOG,
    plans: objectsFor({ posts: () => constant(ALL_POSTS) }),
    query: `{ posts { author { name } } }`,
    variables: {},
  },
  {
    name: "each_loadone",
    sdl: SDL_BLOG,
    plans: objectsFor({ authors: () => constant(AUTHORS) }),
    query: `{ authors { coauthors { name } } }`,
    variables: {},
  },
  {
    name: "empty_list",
    sdl: SDL_BLOG,
    plans: objectsFor({
      author: (_p, args) => loadOne(args.getRaw("id"), LOADERS.loadAuthors),
    }),
    query: `{ author(id: 3) { posts { id } } }`,
    variables: {},
  },
  {
    name: "explicit_error",
    sdl: SDL_BLOG,
    plans: objectsFor({
      boom: () =>
        lambda(constant(0), () => {
          throw new Error("boom");
        }),
    }),
    query: `{ boom }`,
    variables: {},
  },
];
