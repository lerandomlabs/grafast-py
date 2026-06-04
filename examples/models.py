"""Declarative SQLAlchemy models for the ``grafast_demo`` tables.

A demo/test FIXTURE (NOT shipped library code — hence it freely imports SQLAlchemy)
mapping EXACTLY the tables created by :mod:`examples.seed`: ``authors`` -> ``posts``
-> ``comments`` with the FK graph wired by ``relationship``. Used to show that
:func:`grafast_py.pg.resources_from_models` derives the same resource descriptors a
host would otherwise hand-write in :mod:`examples.demo_schema`.

Column declaration order matters: it must equal the hand-declared resources so the
derived ordered column lists compare equal (``authors [id, name]``,
``posts [id, author_id, title]``, ``comments [id, post_id, author_id, body]``).
"""

from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Author(Base):
    __tablename__ = "authors"
    __table_args__ = {"schema": "grafast_demo"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)

    posts: Mapped[list["Post"]] = relationship("Post", back_populates="author")


class Post(Base):
    __tablename__ = "posts"
    __table_args__ = {"schema": "grafast_demo"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    author_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("grafast_demo.authors.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)

    author: Mapped["Author"] = relationship("Author", back_populates="posts")
    comments: Mapped[list["Comment"]] = relationship(
        "Comment", back_populates="post"
    )


class Comment(Base):
    __tablename__ = "comments"
    __table_args__ = {"schema": "grafast_demo"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    post_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("grafast_demo.posts.id"), nullable=False
    )
    author_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("grafast_demo.authors.id"), nullable=False
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)

    author: Mapped["Author"] = relationship("Author")
    post: Mapped["Post"] = relationship("Post", back_populates="comments")


__all__ = ["Base", "Author", "Post", "Comment"]
